"""VLM Prompt Lab — FiftyOne panel plugin.

Architecture
------------
A hybrid modal panel for interactively testing Vision Language Models on
individual samples. The user loads a ``transformers`` image-text-to-text
pipeline once; it stays alive on the GPU across sample navigation so every
inference call is just ``model.generate`` with no reload cost.

The plugin is split into a Python backend (this file) and a React frontend
(``src/VlmPromptLabPanel.tsx``). Python owns data and computation; React
owns the UI. They communicate through panel methods called via
``usePanelEvent`` on the frontend.

Module reimport problem
-----------------------
FiftyOne does a *fresh* ``importlib`` import of this module on **every**
panel-method call — it does not use the cached ``sys.modules[__name__]``
entry. This means plain module-level globals reset to their initial values
on every request. Two mechanisms are used to work around this:

1. ``_persist`` (pipeline singleton)
   A fake ``types.ModuleType`` stored under a private key in
   ``sys.modules``. FiftyOne only ever replaces ``sys.modules[__name__]``,
   so any other key we register persists for the lifetime of the server
   process. The loaded pipeline lives here.

2. Status files (on-disk IPC)
   Two small JSON files written outside the plugin directory carry state
   between calls and between the main request thread and daemon threads:
     - ``_MODEL_STATUS_FILE``  — model-loading progress, ready, or error.
     - ``_STREAM_STATUS_FILE`` — inference streaming, done, or error.
   File writes use a write-to-temp + atomic rename pattern so a concurrent
   reader never sees a partial file.

   The stream file itself (``_STREAM_FILE``) is a plain UTF-8 text file
   that the inference thread appends to token-by-token. React reads new
   bytes from a stored cursor position every 250 ms, giving a live
   streaming effect without a persistent WebSocket.

Why files must live outside the plugin directory
------------------------------------------------
FiftyOne caches the plugin registry keyed on the modification time of the
plugin directory. Writing any file inside the plugin directory changes its
``mtime``, which invalidates the cache and forces a full reimport on the
next request — defeating the ``_persist`` singleton. All runtime files go
to ``~/.fiftyone/vlm_prompt_lab/`` instead.

Panel methods (called from React via ``usePanelEvent``)
-------------------------------------------------------
``load_model``       — validates params, starts a daemon thread to load
                       the pipeline, returns immediately so the UI doesn't
                       block. React polls ``get_model_status`` for progress.
``free_model``       — sets ``_persist.pipe = None``, calls
                       ``torch.cuda.empty_cache()``, clears status files.
``run_inference``    — builds the messages list, clears the stream file,
                       starts an inference daemon thread, returns
                       immediately. React polls ``get_stream_chunk``.
``get_model_status`` — checks ``_persist.pipe`` first (authoritative),
                       falls back to the status file for in-progress loads.
``get_stream_chunk`` — reads new bytes from the stream file since the
                       last cursor position and returns them along with the
                       current done/error state.
``save_to_field``    — appends a structured run record (model ID,
                       prompts, generation params, latency, response)
                       to a ``ListField(DictField)`` on the sample,
                       creating the field if it doesn't exist yet.
"""

import gc
import json
import os
import re
import sys
import threading
import time
import traceback
import types as _types

import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.operators.types as types

# ---------------------------------------------------------------------------
# Runtime file locations
# ---------------------------------------------------------------------------
# All files go to ~/.fiftyone/vlm_prompt_lab/ — NOT inside the plugin
# directory, which would invalidate FiftyOne's plugin-cache mtime check.

_STATUS_DIR = os.path.join(os.path.expanduser("~"), ".fiftyone", "vlm_prompt_lab")

# Tracks model-loading lifecycle: loading_model → ready | load_error
_MODEL_STATUS_FILE = os.path.join(_STATUS_DIR, ".model_status.json")

# Tracks inference lifecycle: streaming → done | inference_error
_STREAM_STATUS_FILE = os.path.join(_STATUS_DIR, ".stream_status.json")

# Append-only UTF-8 file; inference thread writes tokens, React reads chunks
_STREAM_FILE = os.path.join(_STATUS_DIR, ".stream.txt")

# ---------------------------------------------------------------------------
# Pipeline singleton — survives module reimports via sys.modules stash
# ---------------------------------------------------------------------------
# FiftyOne calls importlib.import_module(__name__) on every panel-method
# request, producing a fresh module object each time.  We register our own
# fake module under a *different* key so FiftyOne never overwrites it.
#
# Attributes on _persist:
#   .pipe      — the loaded transformers pipeline, or None
#   .model_id  — the HuggingFace model ID string that was loaded
#   .device    — the device string passed to pipeline() (e.g. "cuda")

_PERSIST_KEY = "vlm_prompt_lab__persist"
if _PERSIST_KEY not in sys.modules:
    _persist = _types.ModuleType(_PERSIST_KEY)
    _persist.pipe = None
    _persist.model_id = None
    _persist.device = None
    sys.modules[_PERSIST_KEY] = _persist
else:
    # A previous import already registered the module; grab the reference.
    _persist = sys.modules[_PERSIST_KEY]

# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------


def _ensure_status_dir() -> None:
    """Create the status directory on first use."""
    os.makedirs(_STATUS_DIR, exist_ok=True)


def _write_json(path: str, data: dict) -> None:
    """Atomically write *data* as JSON.

    Write-to-temp + ``os.replace`` ensures a concurrent reader never sees
    a partial or empty file, even if the process is interrupted mid-write.
    """
    _ensure_status_dir()
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def _read_json(path: str) -> "dict | None":
    """Read a JSON file. Returns ``None`` on any failure (missing, corrupt)."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _remove_file(path: str) -> None:
    """Delete a file, silently ignoring the case where it doesn't exist."""
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def _append_stream(text: str) -> None:
    """Append *text* to the stream file and flush immediately.

    Called from the inference thread for each token chunk yielded by
    ``TextIteratorStreamer``. The explicit flush ensures React's polling
    reads the bytes as soon as they're written rather than waiting for
    the OS buffer to fill.
    """
    _ensure_status_dir()
    with open(_STREAM_FILE, "a", encoding="utf-8") as f:
        f.write(text)
        f.flush()


def _clear_stream() -> None:
    """Delete the stream file and stream-status file before a new run.

    Called at the start of every ``run_inference`` call so React never
    reads stale tokens from a previous inference run.
    """
    _remove_file(_STREAM_FILE)
    _remove_file(_STREAM_STATUS_FILE)


# ---------------------------------------------------------------------------
# VRAM helpers
# ---------------------------------------------------------------------------


def _get_vram_info() -> dict:
    """Return current and total VRAM in GB, or an empty dict if unavailable.

    ``memory_allocated`` reports bytes held by live PyTorch tensors (i.e.
    the model weights). It's slightly lower than ``memory_reserved``, which
    includes the allocator's internal caching pool, but it's the more
    meaningful number for "how much memory is this model using".
    """
    try:
        import torch
        if torch.cuda.is_available():
            used  = torch.cuda.memory_allocated() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            return {"vram_gb": round(used, 1), "total_vram_gb": round(total, 1)}
    except Exception:
        pass
    return {}


# ---------------------------------------------------------------------------
# Model loading thread
# ---------------------------------------------------------------------------


def _load_model_thread(model_id: str, device: str, torch_dtype_str: str) -> None:
    """Load the pipeline and store it in the persist singleton.

    Runs in a daemon thread started by ``VlmPromptLabPanel.load_model``.
    Progress is communicated back to React via ``_MODEL_STATUS_FILE`` so
    ``get_model_status`` polls can surface loading messages without the
    main request thread blocking.

    On success: sets ``_persist.pipe`` and writes ``status="ready"``.
    On failure: ensures ``_persist.pipe`` stays ``None`` and writes
    ``status="load_error"`` with the error message.

    Parameters
    ----------
    model_id : str
        HuggingFace model repo ID, e.g. ``"Qwen/Qwen2-VL-7B-Instruct"``.
    device : str
        PyTorch device string: ``"cuda"``, ``"cpu"``, or ``"mps"``.
    torch_dtype_str : str
        One of ``"bfloat16"``, ``"float16"``, ``"float32"``, ``"auto"``.
    """
    try:
        import torch
        from transformers import pipeline as hf_pipeline

        # Write an in-progress status before the slow pipeline() call so
        # the React polling loop shows a meaningful message immediately.
        _write_json(_MODEL_STATUS_FILE, {
            "status": "loading_model",
            "message": f"Loading {model_id}…",
        })

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16":  torch.float16,
            "float32":  torch.float32,
            "auto":     "auto",
        }
        torch_dtype = dtype_map.get(torch_dtype_str, torch.bfloat16)

        # pipeline() downloads weights on first call and caches to
        # ~/.cache/huggingface/hub. Subsequent calls load from disk.
        pipe = hf_pipeline(
            "image-text-to-text",
            model=model_id,
            device=device,
            torch_dtype=torch_dtype,
        )

        # Store in the persist singleton *before* writing the ready status
        # so get_model_status() never sees "ready" with a None pipe.
        _persist.pipe     = pipe
        _persist.model_id = model_id
        _persist.device   = device

        status = {"status": "ready", "model_id": model_id}
        status.update(_get_vram_info())
        _write_json(_MODEL_STATUS_FILE, status)

    except Exception as exc:
        # Ensure the pipe is never left in a partially-initialised state.
        _persist.pipe = None
        _write_json(_MODEL_STATUS_FILE, {
            "status": "load_error",
            "error": str(exc),
        })
        print(f"[vlm_prompt_lab] model load error:\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# Inference thread
# ---------------------------------------------------------------------------


def _run_inference_thread(image_path: str, messages: list, gen_params: dict) -> None:
    """Run a single inference pass and stream tokens to the stream file.

    Runs in a daemon thread started by ``VlmPromptLabPanel.run_inference``.

    The streaming pipeline works as follows:
    1. A ``TextIteratorStreamer`` is created and attached to the model's
       ``generate`` call, which runs in a *second* inner thread.
    2. This function iterates over the streamer, appending each token chunk
       to ``_STREAM_FILE`` as it's produced.
    3. React polls ``get_stream_chunk`` every 250 ms, reading new bytes
       since its last cursor position, producing a live typing effect.
    4. When ``generate`` finishes, ``_STREAM_STATUS_FILE`` is updated to
       ``status="done"`` with token count and latency. React's next poll
       sees ``done=True`` and stops the polling loop.

    Parameters
    ----------
    image_path : str
        Absolute filesystem path to the current sample's image.
    messages : list
        Chat messages list in the format expected by
        ``tokenizer.apply_chat_template``, already assembled by
        ``run_inference`` (system prompt + user prompt with image token).
    gen_params : dict
        Generation parameters from the React panel, e.g.
        ``{"do_sample": True, "temperature": 0.7, "max_new_tokens": 512}``.
    """
    try:
        from PIL import Image
        from transformers import GenerationConfig, TextIteratorStreamer
        from threading import Thread

        pipe = _persist.pipe
        if pipe is None:
            # Shouldn't happen since run_inference validates first, but
            # guard here in case the model was freed between the check and
            # the thread starting.
            _write_json(_STREAM_STATUS_FILE, {
                "status": "inference_error",
                "error": "Model not loaded.",
            })
            return

        # Write "streaming" status immediately so React's first 250ms poll
        # doesn't see an empty status file and assume inference is done.
        _write_json(_STREAM_STATUS_FILE, {
            "status": "streaming",
            "start_time": time.time(),
        })

        image = Image.open(image_path).convert("RGB")

        # apply_chat_template formats the messages list into the model's
        # expected prompt string (including special tokens like <|im_start|>).
        # add_generation_prompt=True appends the assistant turn-start token
        # so the model knows to generate a response.
        text = pipe.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # pipe.preprocess handles the image+text tokenisation and returns a
        # dict of tensors. We move each tensor to the model's device.
        inputs = pipe.preprocess({"images": image, "text": text})
        inputs = {k: v.to(pipe.device) for k, v in inputs.items() if hasattr(v, "to")}

        # TextIteratorStreamer decouples generation from token-by-token
        # consumption: generate() runs in a background thread and pushes
        # decoded text into a queue; we iterate over the streamer in this
        # thread to drain the queue and write to the stream file.
        streamer = TextIteratorStreamer(
            pipe.tokenizer, skip_special_tokens=True, skip_prompt=True
        )

        # Build a GenerationConfig only with params the caller actually set.
        # Passing None values to GenerationConfig raises validation errors,
        # so we build the kwargs dict conditionally.
        gen_config_kwargs = {}

        max_new_tokens = gen_params.get("max_new_tokens")
        if max_new_tokens is not None:
            gen_config_kwargs["max_new_tokens"] = int(max_new_tokens)

        do_sample = bool(gen_params.get("do_sample", True))
        gen_config_kwargs["do_sample"] = do_sample

        if do_sample:
            # temperature, top_p, top_k are only meaningful when do_sample=True.
            # Passing them with do_sample=False would raise a warning and they'd
            # be silently ignored, so we skip them here as well.
            for key, cast in (("temperature", float), ("top_p", float), ("top_k", int)):
                if gen_params.get(key) is not None:
                    gen_config_kwargs[key] = cast(gen_params[key])

        rep_penalty = gen_params.get("repetition_penalty")
        if rep_penalty is not None:
            gen_config_kwargs["repetition_penalty"] = float(rep_penalty)

        gen_config = GenerationConfig(**gen_config_kwargs)

        import torch
        generator = None
        seed = gen_params.get("seed")
        if seed is not None:
            # A manual seed makes the output reproducible for a given
            # prompt + parameter combination.
            generator = torch.Generator(device=pipe.device).manual_seed(int(seed))

        generation_kwargs = {
            **inputs,
            "generation_config": gen_config,
            "streamer": streamer,
        }
        if generator is not None:
            generation_kwargs["generator"] = generator

        t0 = time.time()

        # Run generate in a separate thread so this thread is free to drain
        # the streamer queue and write tokens to the stream file concurrently.
        thread = Thread(target=pipe.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            _append_stream(new_text)

        # Wait for generate to fully finish before computing final stats.
        thread.join()
        latency_ms = int((time.time() - t0) * 1000)

        # Count output tokens accurately using the tokenizer rather than
        # counting streamer yields (which are text chunks, not tokens).
        token_count = None
        try:
            with open(_STREAM_FILE, "r", encoding="utf-8") as f:
                full_text = f.read()
            token_count = len(
                pipe.tokenizer.encode(full_text, add_special_tokens=False)
            )
        except Exception:
            pass

        _write_json(_STREAM_STATUS_FILE, {
            "status": "done",
            "token_count": token_count,
            "latency_ms": latency_ms,
            # Pass temperature back so the footer can display it even if
            # the user changed the UI slider between Run and completion.
            "temperature": gen_params.get("temperature") if do_sample else None,
        })

    except Exception as exc:
        _write_json(_STREAM_STATUS_FILE, {
            "status": "inference_error",
            "error": str(exc),
        })
        print(f"[vlm_prompt_lab] inference error:\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------


class VlmPromptLabPanel(foo.Panel):
    """Modal panel for interactive VLM prompt testing.

    Displayed in the FiftyOne sample modal sidebar. Lifecycle hooks keep
    the current sample's filepath in panel state; all prompt/model state
    lives on the React side and is unaffected by sample navigation.
    """

    @property
    def config(self):
        return foo.PanelConfig(
            name="vlm_prompt_lab",
            label="VLM Prompt Lab",
            # "modal" surface places this panel in the sample modal sidebar,
            # which is where the user can see the image while prompting.
            surfaces="modal",
            help_markdown=(
                "Load any `image-text-to-text` model, write system/user "
                "prompts, tweak generation parameters, and stream the model "
                "response right in the panel. Save good outputs directly to "
                "a dataset field."
            ),
        )

    # ── Lifecycle hooks ──────────────────────────────────────────────────────

    def on_load(self, ctx):
        """Called once when the panel is first opened in the modal.

        Pushes the current sample filepath and model state to React so the
        UI initialises correctly without waiting for the first poll cycle.
        """
        self._sync_sample(ctx)

        # Check _persist.pipe (the authoritative source) rather than the
        # status file, which may be stale from a previous session.
        if _persist.pipe is not None:
            status = {"status": "ready", "model_id": _persist.model_id or ""}
            status.update(_get_vram_info())
        else:
            status = {"status": "unloaded"}

        # React reads this via props.data.initial_model_status on mount.
        ctx.panel.set_state("initial_model_status", status)

    def on_change_current_sample(self, ctx):
        """Called every time the user navigates to a different sample.

        Only the filepath and sample ID are updated. Prompts, generation
        parameters, and model state are untouched — intentionally, so the
        user can paginate through samples without retyping anything.
        """
        self._sync_sample(ctx)

    def _sync_sample(self, ctx):
        """Push filepath and sample ID to React state.

        React watches ``props.data.image_path`` and clears the output pane
        when it changes, signalling that a new Run is needed for this sample.
        """
        if not ctx.current_sample:
            return
        sample = ctx.dataset[ctx.current_sample]
        ctx.panel.set_state("sample_id", ctx.current_sample)
        ctx.panel.set_state("image_path", sample.filepath)

    # ── Panel methods ────────────────────────────────────────────────────────

    def load_model(self, ctx):
        """Start loading the pipeline in a daemon thread and return immediately.

        React transitions to "loading" state on this response and starts
        polling ``get_model_status`` every second until it sees "ready".

        Parameters (via ctx.params)
        ---------------------------
        model_id : str
            HuggingFace model ID, e.g. ``"Qwen/Qwen2-VL-7B-Instruct"``.
        device : str
            ``"cuda"`` | ``"cpu"`` | ``"mps"``  (default ``"cuda"``).
        torch_dtype : str
            ``"bfloat16"`` | ``"float16"`` | ``"float32"`` | ``"auto"``
            (default ``"bfloat16"``).
        """
        model_id   = ctx.params.get("model_id",    "").strip()
        device     = ctx.params.get("device",      "cuda")
        torch_dtype = ctx.params.get("torch_dtype", "bfloat16")

        if not model_id:
            return {"error": "Model ID is required."}

        # Prevent accidentally starting a second load when one is in flight
        # or a model is already resident on the GPU.
        if _persist.pipe is not None:
            return {"error": "A model is already loaded. Free it first."}

        thread = threading.Thread(
            target=_load_model_thread,
            kwargs=dict(model_id=model_id, device=device, torch_dtype_str=torch_dtype),
            daemon=True,  # exits automatically if the server process exits
        )
        thread.start()
        return {"status": "loading"}

    def free_model(self, ctx):
        """Release the pipeline and free GPU memory.

        Sets ``_persist.pipe = None`` before calling ``empty_cache`` so
        that any concurrent ``get_model_status`` poll that fires in the gap
        sees "unloaded" rather than "ready" with a stale pipe reference.
        """
        if _persist.pipe is None:
            return {"status": "already_free"}

        try:
            import torch
            # Clear the persist references first so no new inference can
            # grab the pipe after we've started tearing it down.
            _persist.pipe     = None
            _persist.model_id = None
            _persist.device   = None
            # empty_cache releases the CUDA allocator's cached blocks back
            # to the OS. gc.collect handles any Python-side cyclic refs.
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as exc:
            return {"error": str(exc)}

        _remove_file(_MODEL_STATUS_FILE)
        _clear_stream()
        return {"status": "freed"}

    def run_inference(self, ctx):
        """Assemble the messages list and start the inference daemon thread.

        Returns immediately with ``status="started"``; React begins polling
        ``get_stream_chunk`` every 250 ms to receive token chunks.

        The messages list is built here (not in the thread) so the thread
        receives a complete, validated payload:
        - System prompt is included only if the user filled it in.
        - The image token ``{"type": "image"}`` must come before the text
          token in the user content list for most VLMs.

        Parameters (via ctx.params)
        ---------------------------
        image_path : str
            Absolute path to the current sample's image file.
        system_prompt : str
            Optional system prompt. Empty string means no system message.
        user_prompt : str
            Required user-facing prompt.
        gen_params : dict
            Generation parameters forwarded to ``_run_inference_thread``.
        """
        if _persist.pipe is None:
            return {"error": "No model loaded."}

        system_prompt = ctx.params.get("system_prompt", "").strip()
        user_prompt   = ctx.params.get("user_prompt",   "").strip()
        gen_params    = ctx.params.get("gen_params",    {})
        image_path    = ctx.params.get("image_path",    "")

        if not user_prompt:
            return {"error": "User prompt is required."}
        if not image_path:
            return {"error": "No image path provided."}
        if not os.path.exists(image_path):
            return {"error": f"Image file not found: {image_path}"}

        # Build the messages list that tokenizer.apply_chat_template expects.
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({
            "role": "user",
            "content": [
                # The image placeholder must appear before the text content
                # so the vision encoder processes it first.
                {"type": "image"},
                {"type": "text", "text": user_prompt},
            ],
        })

        # Clear any previous run's data *before* starting the thread so
        # React's polling loop never reads stale tokens if it fires between
        # this call returning and the thread writing its first chunk.
        _clear_stream()

        thread = threading.Thread(
            target=_run_inference_thread,
            kwargs=dict(
                image_path=image_path,
                messages=messages,
                gen_params=gen_params,
            ),
            daemon=True,
        )
        thread.start()
        return {"status": "started"}

    def get_model_status(self, ctx):
        """Return the current model lifecycle state for React polling.

        Priority order:
        1. If ``_persist.pipe`` is not None, the model is definitely ready
           regardless of what the status file says.
        2. Otherwise read the status file, which the load thread writes to.
        3. If no file exists, the model has never been loaded this session.
        """
        if _persist.pipe is not None:
            status = {"status": "ready", "model_id": _persist.model_id or ""}
            status.update(_get_vram_info())
            return status

        state = _read_json(_MODEL_STATUS_FILE)
        return state if state else {"status": "unloaded"}

    def get_stream_chunk(self, ctx):
        """Return new streamed text since the last cursor position.

        React calls this every 250 ms while inference is running. It passes
        the byte offset it last read to; this method seeks to that position
        and reads whatever the inference thread has appended since then.

        Using byte offsets rather than line/character offsets handles
        multi-byte UTF-8 characters correctly.

        Parameters (via ctx.params)
        ---------------------------
        cursor : int
            Byte offset in ``_STREAM_FILE`` from which to start reading.

        Returns
        -------
        dict with keys:
            text         — new decoded text since cursor (may be empty string)
            cursor       — updated byte offset after reading
            done         — True when inference is complete or errored
            final_status — full status dict when done, else None
        """
        cursor = ctx.params.get("cursor", 0)

        try:
            with open(_STREAM_FILE, "rb") as f:
                f.seek(cursor)
                new_bytes = f.read()
            new_cursor = cursor + len(new_bytes)
            new_text   = new_bytes.decode("utf-8", errors="replace")
        except FileNotFoundError:
            # The stream file doesn't exist yet (inference thread hasn't
            # written its first token) or was cleared. Return empty.
            new_text   = ""
            new_cursor = cursor

        stream_status = _read_json(_STREAM_STATUS_FILE) or {}
        inference_state = stream_status.get("status", "streaming")
        done = inference_state in ("done", "inference_error")

        return {
            "text":         new_text,
            "cursor":       new_cursor,
            "done":         done,
            # Only include final_status when done so React knows to stop
            # polling and can display the token count / latency footer.
            "final_status": stream_status if done else None,
        }

    def save_to_field(self, ctx):
        """Append a structured inference run record to a ListField(DictField).

        Each call appends one dictionary to the list stored in ``field_name``
        on the current sample.  This lets users accumulate multiple runs
        (different models, prompts, or generation configs) on the same sample
        without overwriting earlier results.

        The saved dict contains every piece of information needed to reproduce
        and compare a run:

            {
                "model_id":        str,
                "system_prompt":   str,
                "user_prompt":     str,
                "generation_params": {
                    "do_sample": bool,
                    "temperature": float | None,
                    "max_new_tokens": int,
                    "top_p": float | None,
                    "top_k": int | None,
                    "repetition_penalty": float,
                    "seed": int | None,
                },
                "latency_ms":  int | None,
                "response":    str,
            }

        Parameters (via ctx.params)
        ---------------------------
        field_name : str
            Destination field — will be created as ``ListField(DictField)``
            if it doesn't exist yet.
        sample_id : str
            FiftyOne sample ID of the current sample.
        entry : dict
            The structured run record to append.
        """
        field_name = ctx.params.get("field_name", "").strip()
        sample_id  = ctx.params.get("sample_id",  "")
        entry      = ctx.params.get("entry",       {})

        if not field_name:
            return {"error": "Field name is required."}
        if not sample_id:
            return {"error": "No sample selected."}
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", field_name):
            return {"error": "Invalid field name — use letters, numbers, and underscores only."}
        if not entry:
            return {"error": "No entry data provided."}

        try:
            dataset = ctx.dataset

            # Ensure the field exists as ListField(DictField) so FiftyOne
            # can index and filter individual keys inside each dict.
            if field_name not in dataset.get_field_schema():
                dataset.add_sample_field(
                    field_name,
                    fo.ListField,
                    subfield=fo.DictField,
                )

            sample   = dataset[sample_id]
            existing = sample.get_field(field_name) or []
            existing.append(entry)
            sample[field_name] = existing
            sample.save()

            return {"saved": True, "count": len(existing)}
        except Exception as exc:
            return {"error": str(exc)}

    def get_dataset_fields(self, ctx):
        """Return dataset fields compatible with the save format.

        Scans the current dataset schema and returns only
        ``ListField(DictField)`` fields — the exact type this panel creates
        and appends to.  Private fields and core metadata fields
        (``id``, ``filepath``, ``tags``, ``metadata``) are excluded.
        """
        try:
            schema = ctx.dataset.get_field_schema()
            compatible = []

            for name, field in schema.items():
                # Skip private / internal fields
                if name.startswith("_"):
                    continue
                # Skip core metadata fields that the user should never overwrite
                if name in ("id", "filepath", "tags", "metadata"):
                    continue

                is_list_dict = (
                    isinstance(field, fo.ListField)
                    and isinstance(getattr(field, "field", None), fo.DictField)
                )

                if is_list_dict:
                    compatible.append({"name": name, "kind": "list_dict"})

            return {"fields": compatible}
        except Exception as exc:
            return {"fields": [], "error": str(exc)}

    def render(self, ctx):
        """Return the panel's React component descriptor.

        ``composite_view=True`` tells FiftyOne to look up the component by
        name in the registered plugin components (``PluginComponentType.Component``).
        The keyword arguments (``load_model=self.load_model``, etc.) are
        serialised into ``schema.view.*`` on the React side, giving the
        frontend the operator URI strings it needs to call each method via
        ``usePanelEvent``.
        """
        return types.Property(
            types.Object(),
            view=types.View(
                component="VlmPromptLabPanel",
                composite_view=True,
                load_model=self.load_model,
                free_model=self.free_model,
                run_inference=self.run_inference,
                get_model_status=self.get_model_status,
                get_stream_chunk=self.get_stream_chunk,
                save_to_field=self.save_to_field,
                get_dataset_fields=self.get_dataset_fields,
            ),
        )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register(p):
    """Entry point called by FiftyOne when it loads this plugin."""
    p.register(VlmPromptLabPanel)
