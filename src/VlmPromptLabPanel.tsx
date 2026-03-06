import React, { useCallback, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { usePanelClient } from "./hooks/usePanelClient";
import type { PanelData, PanelSchema } from "./types";

// ---------------------------------------------------------------------------
// Session persistence — survives modal close/reopen and sample navigation
// ---------------------------------------------------------------------------
// The panel can unmount and remount when the user navigates away from the
// modal and returns. sessionStorage is the right scope here: it persists
// for the lifetime of the browser tab but clears when the tab is closed,
// which is the right lifecycle for a prompt-testing session.

const SESSION_KEY = "vlmPromptLab:config";

interface PersistedConfig {
  modelId:        string;
  systemPrompt:   string;
  userPrompt:     string;
  doSample:       boolean;
  temperature:    string;
  maxNewTokens:   string;
  topP:           string;
  topK:           string;
  repPenalty:     string;
  seed:           string;
  fieldName:      string;
  genParamsOpen:  boolean;
  outputHeight:   number;
}

const DEFAULT_CONFIG: PersistedConfig = {
  modelId:        "",
  systemPrompt:   "",
  userPrompt:     "",
  doSample:       true,
  temperature:    "0.7",
  maxNewTokens:   "512",
  topP:           "0.9",
  topK:           "50",
  repPenalty:     "1.0",
  seed:           "",
  fieldName:      "",
  genParamsOpen:  false,   // collapsed by default to keep config compact
  outputHeight:   220,
};

function loadConfig(): PersistedConfig {
  try {
    const raw = sessionStorage.getItem(SESSION_KEY);
    if (raw) return { ...DEFAULT_CONFIG, ...JSON.parse(raw) };
  } catch { /* sessionStorage unavailable */ }
  return DEFAULT_CONFIG;
}

function saveConfig(cfg: PersistedConfig): void {
  try { sessionStorage.setItem(SESSION_KEY, JSON.stringify(cfg)); } catch {}
}

// ---------------------------------------------------------------------------
// Per-sample output cache — survives sample navigation, modal close/reopen,
// and panel close/reopen (all within the same browser tab).
//
// Keyed by FiftyOne sample ID so navigating back to a sample restores the
// exact output that was generated for it.  Capped at 50 entries so it
// doesn't grow unbounded over a long session.
// ---------------------------------------------------------------------------

const OUTPUT_CACHE_KEY = "vlmPromptLab:sampleOutputs";

interface CachedOutput {
  streamText: string;
  tokenCount: number | null;
  latencyMs:  number | null;
  activeTemp: number | null;
}

function getCachedOutput(sampleId: string): CachedOutput | null {
  try {
    const raw = sessionStorage.getItem(OUTPUT_CACHE_KEY);
    if (!raw) return null;
    return (JSON.parse(raw) as Record<string, CachedOutput>)[sampleId] ?? null;
  } catch { return null; }
}

function setCachedOutput(sampleId: string, output: CachedOutput): void {
  try {
    const raw   = sessionStorage.getItem(OUTPUT_CACHE_KEY);
    const cache = raw ? (JSON.parse(raw) as Record<string, CachedOutput>) : {};
    cache[sampleId] = output;
    // Evict oldest entry once the cache exceeds 50 samples.
    const keys = Object.keys(cache);
    if (keys.length > 50) delete cache[keys[0]];
    sessionStorage.setItem(OUTPUT_CACHE_KEY, JSON.stringify(cache));
  } catch {}
}

function clearCachedOutput(sampleId: string): void {
  try {
    const raw = sessionStorage.getItem(OUTPUT_CACHE_KEY);
    if (!raw) return;
    const cache = JSON.parse(raw) as Record<string, CachedOutput>;
    delete cache[sampleId];
    sessionStorage.setItem(OUTPUT_CACHE_KEY, JSON.stringify(cache));
  } catch {}
}

// ---------------------------------------------------------------------------
// Keyframes — injected once into the document head
// ---------------------------------------------------------------------------

let _spinInjected = false;
function ensureSpinKeyframe() {
  if (_spinInjected) return;
  _spinInjected = true;
  const el = document.createElement("style");
  el.textContent = `
    @keyframes vlmSpin  { to { transform: rotate(360deg); } }
    @keyframes vlmBlink { 0%,100%{opacity:1} 50%{opacity:0} }
  `;
  document.head.appendChild(el);
}

// Inject scoped styles for rendered markdown output.
// All rules are nested under .vlm-md so they can't leak into the wider app.
// Colours reference FiftyOne CSS variables so the output matches the theme.
let _mdStylesInjected = false;
function ensureMarkdownStyles() {
  if (_mdStylesInjected) return;
  _mdStylesInjected = true;
  const el = document.createElement("style");
  el.textContent = `
    .vlm-md { font-family: var(--fo-fontFamily-body); font-size: 13px; line-height: 1.7; color: var(--fo-palette-text-primary); }
    .vlm-md p  { margin: 0 0 10px; }
    .vlm-md p:last-child { margin-bottom: 0; }
    .vlm-md h1,.vlm-md h2,.vlm-md h3,.vlm-md h4,.vlm-md h5,.vlm-md h6 {
      margin: 14px 0 6px; font-weight: 600; line-height: 1.3;
      color: var(--fo-palette-text-primary);
    }
    .vlm-md h1 { font-size: 18px; }
    .vlm-md h2 { font-size: 16px; }
    .vlm-md h3 { font-size: 14px; }
    .vlm-md h4,.vlm-md h5,.vlm-md h6 { font-size: 13px; }
    .vlm-md ul,.vlm-md ol { margin: 0 0 10px; padding-left: 20px; }
    .vlm-md li { margin-bottom: 3px; }
    .vlm-md code {
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 12px;
      background: var(--fo-palette-background-level2);
      color: var(--fo-palette-primary-main);
      padding: 1px 5px;
      border-radius: 3px;
    }
    .vlm-md pre {
      background: var(--fo-palette-background-level2);
      border: 1px solid var(--fo-palette-divider);
      border-radius: 4px;
      padding: 10px 12px;
      overflow-x: auto;
      margin: 0 0 10px;
    }
    .vlm-md pre code {
      background: none;
      padding: 0;
      color: var(--fo-palette-text-primary);
      font-size: 12px;
    }
    .vlm-md blockquote {
      margin: 0 0 10px;
      padding: 4px 12px;
      border-left: 3px solid var(--fo-palette-primary-main);
      color: var(--fo-palette-text-secondary);
    }
    .vlm-md table {
      border-collapse: collapse; width: 100%; margin-bottom: 10px; font-size: 12px;
    }
    .vlm-md th,.vlm-md td {
      border: 1px solid var(--fo-palette-divider);
      padding: 5px 8px; text-align: left;
    }
    .vlm-md th { background: var(--fo-palette-background-level3); font-weight: 600; }
    .vlm-md a  { color: var(--fo-palette-primary-main); }
    .vlm-md hr { border: none; border-top: 1px solid var(--fo-palette-divider); margin: 10px 0; }
    .vlm-md strong { font-weight: 600; }
    .vlm-md em    { font-style: italic; }
  `;
  document.head.appendChild(el);
}

// ---------------------------------------------------------------------------
// Design tokens — pulled from FiftyOne's own CSS custom properties
// ---------------------------------------------------------------------------
// These variables are set by the FiftyOne app on :root, so our panel
// automatically inherits the correct values for whatever theme is active.

const V = {
  // Backgrounds
  bgBody:   "var(--fo-palette-background-body)",
  bgLevel2: "var(--fo-palette-background-level2)",
  bgLevel3: "var(--fo-palette-background-level3)",
  bgButton: "var(--fo-palette-background-button)",
  // Borders
  divider:  "var(--fo-palette-divider)",
  softBorder: "var(--fo-palette-neutral-softBorder)",
  // Text
  text:        "var(--fo-palette-text-primary)",
  textMuted:   "var(--fo-palette-text-secondary)",
  textDim:     "var(--fo-palette-text-tertiary)",
  textInvert:  "var(--fo-palette-text-invert)",
  // Primary (FiftyOne orange)
  primary:     "var(--fo-palette-primary-main)",
  primaryBorder: "var(--fo-palette-primary-plainBorder)",
  // Non-variable fallbacks (no semantic CSS var exists for these)
  green:  "#4caf81",
  greenBg: "#0b1a0b",
  greenBorder: "#1a3a1a",
  blue:   "#5b9cf6",
  blueBg: "#111824",
  blueBorder: "#1e2e40",
  red:    "#e08080",
  redBg:  "#2a0e0e",
};

// FiftyOne's body font — same typeface as the rest of the app.
const FONT = "var(--fo-fontFamily-body)";

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

const S: Record<string, React.CSSProperties> = {

  // ── Root — inherits the FiftyOne app's background and text colours ──
  root: {
    display:       "flex",
    flexDirection: "column",
    height:        "100%",
    overflow:      "hidden",
    fontFamily:    FONT,
    fontSize:      13,
    color:         V.text,
    background:    V.bgBody,
    boxSizing:     "border-box",
  },

  // ── Top section (config) ──
  // flex:1 so it fills whatever space isn't claimed by the output area.
  // overflowY:auto lets it scroll when the content (especially if gen-params
  // accordion is open) exceeds the available height.
  topSection: {
    flex:          1,
    overflowY:     "auto",
    minHeight:     0,
    padding:       "10px 12px",
    display:       "flex",
    flexDirection: "column",
    gap:           10,
  },

  // ── Prompt bar — fixed at the bottom of the panel ──
  // Two rows: (1) save-field name + Save button, (2) prompt textarea + Send.
  promptBar: {
    flexShrink:    0,
    padding:       "8px 10px",
    borderTop:     `1px solid ${V.divider}`,
    background:    V.bgLevel2,
    display:       "flex",
    flexDirection: "column",
    gap:           6,
  },

  // User prompt textarea — vertically resizable by the user
  promptInput: {
    flex:         1,
    background:   V.bgBody,
    color:        V.text,
    border:       `1px solid ${V.divider}`,
    borderRadius: 4,
    padding:      "7px 10px",
    fontSize:     13,
    outline:      "none",
    resize:       "vertical" as const,
    fontFamily:   FONT,
    lineHeight:   1.5,
    boxSizing:    "border-box" as const,
    minHeight:    38,
    overflow:     "auto",
  },

  promptInputDisabled: {
    flex:         1,
    background:   V.bgBody,
    color:        V.textDim,
    border:       `1px solid ${V.softBorder}`,
    borderRadius: 4,
    padding:      "7px 10px",
    fontSize:     13,
    outline:      "none",
    resize:       "none" as const,
    fontFamily:   FONT,
    lineHeight:   1.5,
    boxSizing:    "border-box" as const,
    minHeight:    38,
    overflow:     "auto",
    cursor:       "not-allowed",
  },

  // Drag handle — lets the user resize the output area.
  // Sits between runBar and bottomSection; cursor changes to ns-resize.
  resizeHandle: {
    flexShrink:     0,
    height:         8,
    cursor:         "ns-resize",
    background:     V.bgLevel3,
    display:        "flex",
    alignItems:     "center",
    justifyContent: "center",
    userSelect:     "none" as const,
    borderBottom:   `1px solid ${V.divider}`,
  },

  resizeDots: {
    color:        V.textDim,
    fontSize:     10,
    letterSpacing: "2px",
    lineHeight:   1,
    pointerEvents: "none" as const,
  },

  // ── Accordion — generation parameters ──
  accordionHeader: {
    display:    "flex",
    alignItems: "center",
    gap:        5,
    cursor:     "pointer",
    userSelect: "none" as const,
    padding:    "2px 0",
  },

  accordionChevron: {
    fontSize:      9,
    color:         V.textMuted,
    transition:    "transform 0.18s ease",
    lineHeight:    1,
    verticalAlign: "middle",
    display:       "inline-flex",
    alignItems:    "center",
  },

  // ── Bottom section (output + save) ──
  // Height is controlled by the user via the resize handle; stored in state.
  bottomSection: {
    flexShrink:    0,
    display:       "flex",
    flexDirection: "column",
    minHeight:     80,
    overflow:      "hidden",
  },

  outputArea: {
    flex:      1,
    overflowY: "auto",
    padding:   "10px 12px",
    minHeight: 0,
  },

  outputEmpty: {
    height:         "100%",
    display:        "flex",
    alignItems:     "center",
    justifyContent: "center",
    color:          V.textMuted,
    fontSize:       12,
    textAlign:      "center",
    padding:        24,
    fontFamily:     FONT,
  },

  outputText: {
    margin:   0,
    wordBreak: "break-word",
  },

  outputLoading: {
    display:     "flex",
    alignItems:  "center",
    gap:         8,
    color:       V.textMuted,
    fontSize:    12,
    fontFamily:  FONT,
    padding:     "4px 0",
  },

  // Blinking text cursor appended to the end of streaming output
  cursor: {
    display:        "inline-block",
    width:          2,
    height:         "1em",
    background:     V.primary,
    marginLeft:     1,
    verticalAlign:  "text-bottom",
    animation:      "vlmBlink 1s step-end infinite",
  } as React.CSSProperties,

  outputFooter: {
    flexShrink:  0,
    padding:     "4px 12px",
    borderTop:   `1px solid ${V.softBorder}`,
    display:     "flex",
    gap:         12,
    alignItems:  "center",
    color:       V.textMuted,
    fontSize:    11,
    fontFamily:  FONT,
  },


  // ── Form controls ──
  sectionLabel: {
    fontSize:      11,
    fontWeight:    600,
    color:         V.textMuted,
    textTransform: "uppercase" as const,
    letterSpacing: "0.05em",
    marginBottom:  5,
    fontFamily:    FONT,
  },

  input: {
    width:        "100%",
    background:   V.bgLevel2,
    color:        V.text,
    border:       `1px solid ${V.divider}`,
    borderRadius: 4,
    padding:      "5px 8px",
    fontSize:     12,
    outline:      "none",
    boxSizing:    "border-box" as const,
    fontFamily:   FONT,
  },

  inputLocked: {
    width:        "100%",
    background:   V.bgBody,
    color:        V.textDim,
    border:       `1px solid ${V.softBorder}`,
    borderRadius: 4,
    padding:      "5px 8px",
    fontSize:     12,
    outline:      "none",
    cursor:       "not-allowed",
    boxSizing:    "border-box" as const,
    fontFamily:   FONT,
  },

  textarea: {
    width:        "100%",
    background:   V.bgLevel2,
    color:        V.text,
    border:       `1px solid ${V.divider}`,
    borderRadius: 4,
    padding:      "6px 8px",
    fontSize:     12,
    outline:      "none",
    resize:       "vertical" as const,
    fontFamily:   FONT,
    lineHeight:   1.5,
    boxSizing:    "border-box" as const,
    minHeight:    54,
  },

  select: {
    flex:         1,
    background:   V.bgLevel2,
    color:        V.text,
    border:       `1px solid ${V.divider}`,
    borderRadius: 4,
    padding:      "4px 6px",
    fontSize:     12,
    outline:      "none",
    fontFamily:   FONT,
  },

  selectLocked: {
    flex:         1,
    background:   V.bgBody,
    color:        V.textDim,
    border:       `1px solid ${V.softBorder}`,
    borderRadius: 4,
    padding:      "4px 6px",
    fontSize:     12,
    cursor:       "not-allowed",
    fontFamily:   FONT,
  },

  // Gen params use a 3-column grid to make use of the full panel width
  paramGrid: {
    display:             "grid",
    gridTemplateColumns: "1fr 1fr 1fr",
    gap:                 "6px 8px",
  },

  paramCell: {
    display:       "flex",
    flexDirection: "column" as const,
    gap:           3,
  },

  paramLabel: {
    fontSize:   10,
    color:      V.textMuted,
    fontFamily: FONT,
  },

  paramLabelDim: {
    fontSize:   10,
    color:      V.textDim,
    fontFamily: FONT,
  },

  numInput: {
    width:        "100%",
    background:   V.bgLevel2,
    color:        V.text,
    border:       `1px solid ${V.divider}`,
    borderRadius: 4,
    padding:      "4px 6px",
    fontSize:     12,
    outline:      "none",
    boxSizing:    "border-box" as const,
    fontFamily:   FONT,
  },

  numInputDisabled: {
    width:        "100%",
    background:   V.bgBody,
    color:        V.textDim,
    border:       `1px solid ${V.softBorder}`,
    borderRadius: 4,
    padding:      "4px 6px",
    fontSize:     12,
    cursor:       "not-allowed",
    boxSizing:    "border-box" as const,
    fontFamily:   FONT,
  },

  toggleRow: {
    display:    "flex",
    alignItems: "center",
    gap:        7,
    marginBottom: 6,
  },

  toggleLabel: {
    fontSize:   12,
    color:      V.textMuted,
    cursor:     "pointer",
    userSelect: "none" as const,
    fontFamily: FONT,
  },

  // ── Buttons — primary actions use FiftyOne's orange accent colour ──
  btnLoad: {
    width:        "100%",
    background:   V.primary,
    color:        V.textInvert,
    border:       `1px solid ${V.primaryBorder}`,
    borderRadius: 4,
    padding:      "6px 12px",
    fontSize:     12,
    cursor:       "pointer",
    fontWeight:   600,
    fontFamily:   FONT,
  },

  // Secondary/small buttons use the FiftyOne button background token
  btnSmallDisabled: {
    background:   V.bgBody,
    color:        V.textDim,
    border:       `1px solid ${V.softBorder}`,
    borderRadius: 4,
    padding:      "4px 10px",
    fontSize:     12,
    cursor:       "not-allowed",
    flexShrink:   0,
    whiteSpace:   "nowrap" as const,
    fontFamily:   FONT,
  },

  btnDanger: {
    background:     "none",
    color:          V.red,
    border:         "none",
    cursor:         "pointer",
    fontSize:       11,
    padding:        "2px 4px",
    textDecoration: "underline",
    flexShrink:     0,
    fontFamily:     FONT,
  },

  // ── Status / feedback ──
  statusReady: {
    display:      "flex",
    alignItems:   "center",
    gap:          8,
    padding:      "5px 8px",
    background:   V.greenBg,
    border:       `1px solid ${V.greenBorder}`,
    borderRadius: 4,
    fontSize:     11,
    fontFamily:   FONT,
  },

  statusDot: {
    width:        7,
    height:       7,
    borderRadius: "50%",
    background:   V.green,
    flexShrink:   0,
  },

  statusText: {
    flex:         1,
    color:        V.green,
    overflow:     "hidden",
    textOverflow: "ellipsis",
    whiteSpace:   "nowrap" as const,
    fontSize:     11,
    fontFamily:   FONT,
  },

  statusVram: {
    color:      V.textMuted,
    fontSize:   10,
    flexShrink: 0,
    fontFamily: FONT,
  },

  statusLoading: {
    display:      "flex",
    alignItems:   "center",
    gap:          8,
    padding:      "5px 8px",
    background:   V.blueBg,
    border:       `1px solid ${V.blueBorder}`,
    borderRadius: 4,
    fontSize:     11,
    color:        V.blue,
    fontFamily:   FONT,
  },

  spinner: {
    width:        12,
    height:       12,
    border:       `2px solid ${V.divider}`,
    borderTop:    `2px solid ${V.primary}`,
    borderRadius: "50%",
    display:      "inline-block",
    animation:    "vlmSpin 0.7s linear infinite",
    flexShrink:   0,
  } as React.CSSProperties,

  errorBox: {
    padding:      "6px 8px",
    background:   V.redBg,
    color:        V.red,
    borderRadius: 4,
    fontSize:     11,
    lineHeight:   1.4,
    fontFamily:   FONT,
  },

  divider: {
    borderTop: `1px solid ${V.softBorder}`,
  },

  // ── Output footer ──
  // Field-name combobox in the output footer — transparent so it blends with
  // the footer bar background; picks up the active field via a <datalist>.
  footerFieldInput: {
    marginLeft:   "auto",
    width:        180,
    background:   "transparent",
    color:        V.text,
    border:       `1px solid ${V.softBorder}`,
    borderRadius: 3,
    padding:      "1px 6px",
    fontSize:     11,
    outline:      "none",
    fontFamily:   FONT,
  } as React.CSSProperties,

  // Save icon button base — dynamic cursor/opacity/color stay inline.
  saveIconBtn: {
    background:  "none",
    border:      "none",
    padding:     "0 2px",
    fontSize:    14,
    lineHeight:  1,
    flexShrink:  0,
  } as React.CSSProperties,

};

// ---------------------------------------------------------------------------
// Component-level types
// ---------------------------------------------------------------------------

type ModelState     = "unloaded" | "loading" | "ready" | "error";
type InferenceState = "idle" | "running" | "done" | "error";

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface VlmPromptLabPanelProps {
  data?:   PanelData;
  schema?: PanelSchema;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const VlmPromptLabPanel: React.FC<VlmPromptLabPanelProps> = ({ data, schema }) => {
  const uris = {
    load_model:          schema?.view?.load_model          ?? "",
    free_model:          schema?.view?.free_model          ?? "",
    run_inference:       schema?.view?.run_inference       ?? "",
    get_model_status:    schema?.view?.get_model_status    ?? "",
    get_stream_chunk:    schema?.view?.get_stream_chunk    ?? "",
    save_to_field:       schema?.view?.save_to_field       ?? "",
    get_dataset_fields:  schema?.view?.get_dataset_fields  ?? "",
  };

  const { loadModel, freeModel, runInference, getModelStatus, getStreamChunk,
          saveToField, getDatasetFields } =
    usePanelClient(uris);

  // Restore config from sessionStorage so state survives modal close/reopen.
  // Spread into a local so React sees stable initial values (no re-renders).
  const _cfg = loadConfig();

  // ── Model setup inputs — initialised from persisted config ──────────────
  const [modelId,    setModelId]    = useState(_cfg.modelId);
  const [device,     setDevice]     = useState("cuda");
  const [torchDtype, setTorchDtype] = useState("bfloat16");

  // ── Model state ─────────────────────────────────────────────────────────
  const [modelState,     setModelState]     = useState<ModelState>("unloaded");
  const [loadedModelId,  setLoadedModelId]  = useState("");
  const [vramUsed,       setVramUsed]       = useState<number | null>(null);
  const [vramTotal,      setVramTotal]      = useState<number | null>(null);
  const [modelError,     setModelError]     = useState<string | null>(null);

  // ── Prompts — initialised from persisted config ──────────────────────────
  const [systemPrompt, setSystemPrompt] = useState(_cfg.systemPrompt);
  const [userPrompt,   setUserPrompt]   = useState(_cfg.userPrompt);

  // ── Generation parameters — initialised from persisted config ────────────
  const [doSample,     setDoSample]     = useState(_cfg.doSample);
  const [temperature,  setTemperature]  = useState(_cfg.temperature);
  const [maxNewTokens, setMaxNewTokens] = useState(_cfg.maxNewTokens);
  const [topP,         setTopP]         = useState(_cfg.topP);
  const [topK,         setTopK]         = useState(_cfg.topK);
  const [repPenalty,   setRepPenalty]   = useState(_cfg.repPenalty);
  const [seed,         setSeed]         = useState(_cfg.seed);

  // ── Inference state ──────────────────────────────────────────────────────
  const [inferenceState,  setInferenceState]  = useState<InferenceState>("idle");
  const [streamText,      setStreamText]      = useState("");
  const [tokenCount,      setTokenCount]      = useState<number | null>(null);
  const [latencyMs,       setLatencyMs]       = useState<number | null>(null);
  const [activeTemp,      setActiveTemp]      = useState<number | null>(null);
  const [inferenceError,  setInferenceError]  = useState<string | null>(null);
  const streamCursorRef                       = useRef(0);

  // ── Save state — field name also persisted ───────────────────────────────
  const [fieldName,   setFieldName]   = useState(_cfg.fieldName);
  const [saving,      setSaving]      = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);
  const [saveError,   setSaveError]   = useState<string | null>(null);

  // ── UI layout state — both persisted ─────────────────────────────────────
  const [genParamsOpen, setGenParamsOpen] = useState(_cfg.genParamsOpen);
  const [outputHeight,  setOutputHeight]  = useState(_cfg.outputHeight);

  // ── Sample context from Python lifecycle ─────────────────────────────────
  const [sampleId,  setSampleId]  = useState("");
  const [imagePath, setImagePath] = useState("");

  const outputScrollRef  = useRef<HTMLDivElement>(null);
  const prevImagePathRef = useRef("");
  const promptInputRef   = useRef<HTMLTextAreaElement>(null);

  // Filled in after handleRun is defined so the ref is always current.
  const handleRunRef = useRef<() => void>(() => {});

  // Enter = send (if ready); Shift+Enter = newline.
  const handlePromptKeyDown = useCallback((e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleRunRef.current();
    }
  }, []);

  // ── Resize drag logic ────────────────────────────────────────────────────
  // Refs so the mousemove handler always sees the latest values without
  // needing to be re-registered every render.
  const isDraggingRef        = useRef(false);
  const dragStartYRef        = useRef(0);
  const dragStartHeightRef   = useRef(0);

  const handleResizeMouseDown = useCallback((e: React.MouseEvent) => {
    isDraggingRef.current      = true;
    dragStartYRef.current      = e.clientY;
    dragStartHeightRef.current = outputHeight;
    e.preventDefault();
  }, [outputHeight]);

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (!isDraggingRef.current) return;
      // Dragging UP (negative delta) → taller output; DOWN → shorter.
      const delta  = dragStartYRef.current - e.clientY;
      const newH   = Math.max(80, Math.min(700, dragStartHeightRef.current + delta));
      setOutputHeight(newH);
    };
    const onUp = () => { isDraggingRef.current = false; };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup",   onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup",   onUp);
    };
  }, []); // eslint-disable-line

  // ── Init ─────────────────────────────────────────────────────────────────
  useEffect(() => { ensureSpinKeyframe(); ensureMarkdownStyles(); }, []);

  // Persist config to sessionStorage whenever any of these values change.
  // This runs after every relevant state update, so the latest values are
  // always in storage before the component could unmount.
  useEffect(() => {
    saveConfig({ modelId, systemPrompt, userPrompt, doSample, temperature,
                 maxNewTokens, topP, topK, repPenalty, seed, fieldName,
                 genParamsOpen, outputHeight });
  }, [modelId, systemPrompt, userPrompt, doSample, temperature,
      maxNewTokens, topP, topK, repPenalty, seed, fieldName,
      genParamsOpen, outputHeight]); // eslint-disable-line

  // Initialise model state from Python's on_load so the UI is correct on
  // first render without waiting for the first get_model_status poll.
  const initDone = useRef(false);
  useEffect(() => {
    if (initDone.current) return;
    const init = data?.initial_model_status;
    if (!init) return;
    initDone.current = true;
    if (init.status === "ready") {
      setModelState("ready");
      setLoadedModelId(init.model_id ?? "");
      if (init.vram_gb != null) {
        setVramUsed(init.vram_gb);
        setVramTotal(init.total_vram_gb ?? null);
      }
    }
  }, [data?.initial_model_status]); // eslint-disable-line

  // Sync sample when the user navigates, closes/reopens the modal, or
  // closes/reopens the panel.  prevImagePathRef starts empty on every mount,
  // so this always fires on the first data push from Python's on_load —
  // which means the cache-restore logic also runs on panel reopen. ✓
  useEffect(() => {
    const newPath = data?.image_path ?? "";
    const newId   = data?.sample_id  ?? "";
    if (!newPath || !newId) return;
    if (newPath === prevImagePathRef.current) return;
    prevImagePathRef.current = newPath;
    setImagePath(newPath);
    setSampleId(newId);
    setSaveSuccess(false);
    setSaveError(null);
    streamCursorRef.current = 0;

    // Restore the last inference result for this sample if one exists,
    // otherwise reset to the idle empty state.
    const cached = getCachedOutput(newId);
    if (cached) {
      setStreamText(cached.streamText);
      setTokenCount(cached.tokenCount);
      setLatencyMs(cached.latencyMs);
      setActiveTemp(cached.activeTemp);
      setInferenceState("done");
      setInferenceError(null);
    } else {
      setStreamText("");
      setInferenceState("idle");
      setTokenCount(null);
      setLatencyMs(null);
      setActiveTemp(null);
      setInferenceError(null);
    }
  }, [data?.image_path, data?.sample_id]); // eslint-disable-line

  // Poll get_model_status once per second while the model is loading.
  useEffect(() => {
    if (modelState !== "loading") return;
    const id = setInterval(() => {
      getModelStatus()
        .then((r) => {
          if (r.status === "ready") {
            clearInterval(id);
            setModelState("ready");
            setLoadedModelId(r.model_id ?? "");
            if (r.vram_gb != null) { setVramUsed(r.vram_gb); setVramTotal(r.total_vram_gb ?? null); }
          } else if (r.status === "load_error") {
            clearInterval(id);
            setModelState("error");
            setModelError(r.error ?? "Failed to load model.");
          }
        })
        .catch(() => {});
    }, 1000);
    return () => clearInterval(id);
  }, [modelState]); // eslint-disable-line

  // Poll get_stream_chunk every 250 ms while inference is running.
  // streamCursorRef is a ref (not state) to avoid stale closures inside
  // the setInterval callback without triggering re-renders on every chunk.
  useEffect(() => {
    if (inferenceState !== "running") return;
    const id = setInterval(() => {
      getStreamChunk(streamCursorRef.current)
        .then((r) => {
          if (r.text) {
            setStreamText((prev) => prev + r.text);
            streamCursorRef.current = r.cursor;
          }
          if (r.done) {
            clearInterval(id);
            const fs = r.final_status;
            if (fs?.status === "inference_error") {
              setInferenceError(fs.error ?? "Inference failed.");
              setInferenceState("error");
            } else {
              setTokenCount(fs?.token_count ?? null);
              setLatencyMs(fs?.latency_ms  ?? null);
              setActiveTemp(fs?.temperature ?? null);
              setInferenceState("done");
            }
          }
        })
        .catch(() => {});
    }, 250);
    return () => clearInterval(id);
  }, [inferenceState]); // eslint-disable-line

  // Auto-scroll the output area as tokens stream in.
  useEffect(() => {
    const el = outputScrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [streamText]);

  // Persist completed output to the per-sample cache as soon as inference
  // finishes so navigating away and back restores it automatically.
  useEffect(() => {
    if (inferenceState !== "done" || !streamText || !sampleId) return;
    setCachedOutput(sampleId, { streamText, tokenCount, latencyMs, activeTemp });
  }, [inferenceState]); // eslint-disable-line

  // Auto-dismiss save success message after 3 seconds.
  useEffect(() => {
    if (!saveSuccess) return;
    const t = setTimeout(() => setSaveSuccess(false), 3000);
    return () => clearTimeout(t);
  }, [saveSuccess]);

  // ── Handlers ──────────────────────────────────────────────────────────────

  const handleLoad = useCallback(async () => {
    if (!modelId.trim() || modelState === "loading") return;
    setModelError(null);
    setModelState("loading");
    try {
      await loadModel({ model_id: modelId.trim(), device, torch_dtype: torchDtype });
    } catch (e: any) {
      setModelState("error");
      setModelError(e?.message ?? "Failed to start model loading.");
    }
  }, [modelId, device, torchDtype, modelState, loadModel]);

  const handleFree = useCallback(async () => {
    try { await freeModel(); } catch { /* best effort */ }
    setModelState("unloaded");
    setLoadedModelId(""); setVramUsed(null); setVramTotal(null); setModelError(null);
    setStreamText(""); setInferenceState("idle");
    setTokenCount(null); setLatencyMs(null); setActiveTemp(null); setInferenceError(null);
    streamCursorRef.current = 0;
  }, [freeModel]);

  const handleRun = useCallback(async () => {
    if (!imagePath || !userPrompt.trim() || modelState !== "ready" || inferenceState === "running") return;
    setStreamText(""); setInferenceState("running");
    setInferenceError(null); setTokenCount(null); setLatencyMs(null); setActiveTemp(null);
    setSaveSuccess(false); setSaveError(null);
    streamCursorRef.current = 0;
    // Clear the stale cache entry so a new run replaces the old output.
    if (sampleId) clearCachedOutput(sampleId);

    const gen_params: Record<string, any> = {
      do_sample:      doSample,
      max_new_tokens: maxNewTokens !== "" ? Number(maxNewTokens) : 512,
    };
    if (doSample) {
      if (temperature !== "") gen_params.temperature = Number(temperature);
      if (topP !== "")        gen_params.top_p       = Number(topP);
      if (topK !== "")        gen_params.top_k       = Number(topK);
    }
    if (repPenalty !== "") gen_params.repetition_penalty = Number(repPenalty);
    if (seed !== "")       gen_params.seed               = Number(seed);

    try {
      await runInference({
        image_path:    imagePath,
        system_prompt: systemPrompt,
        user_prompt:   userPrompt.trim(),
        gen_params,
      });
    } catch (e: any) {
      setInferenceError(e?.message ?? "Inference failed.");
      setInferenceState("error");
    }
  }, [
    imagePath, userPrompt, systemPrompt, modelState, inferenceState,
    doSample, temperature, maxNewTokens, topP, topK, repPenalty, seed,
    runInference,
  ]);

  // Keep the ref in sync so handlePromptKeyDown can always call the latest version.
  useEffect(() => { handleRunRef.current = handleRun; }, [handleRun]);

  // ── Dataset field suggestions ─────────────────────────────────────────────
  // Only ListField(DictField) fields are returned — the exact type this panel
  // creates. Fetched lazily on first focus so the list always reflects the
  // current dataset schema without periodic polling.
  const [compatibleFields, setCompatibleFields] = useState<string[]>([]);

  const fetchFields = useCallback(() => {
    if (!uris.get_dataset_fields) return;
    getDatasetFields()
      .then(r => setCompatibleFields(r.fields.map(f => f.name)))
      .catch(() => {});
  }, [getDatasetFields, uris.get_dataset_fields]); // eslint-disable-line

  const handleSave = useCallback(async () => {
    if (!streamText || !fieldName.trim() || !sampleId || saving) return;
    setSaving(true); setSaveSuccess(false); setSaveError(null);
    try {
      // Build the structured generation_params dict, omitting sampling-only
      // keys when do_sample is off so the saved record is self-consistent.
      const generation_params: Record<string, any> = {
        do_sample:           doSample,
        max_new_tokens:      maxNewTokens !== "" ? Number(maxNewTokens) : null,
        repetition_penalty:  repPenalty   !== "" ? Number(repPenalty)   : null,
        seed:                seed         !== "" ? Number(seed)         : null,
      };
      if (doSample) {
        generation_params.temperature = temperature !== "" ? Number(temperature) : null;
        generation_params.top_p       = topP        !== "" ? Number(topP)        : null;
        generation_params.top_k       = topK        !== "" ? Number(topK)        : null;
      }

      await saveToField({
        field_name: fieldName.trim(),
        sample_id:  sampleId,
        entry: {
          model_id:          loadedModelId,
          system_prompt:     systemPrompt,
          user_prompt:       userPrompt,
          generation_params,
          latency_ms:        latencyMs,
          response:          streamText,
        },
      });
      setSaveSuccess(true);
    } catch (e: any) {
      setSaveError(e?.message ?? "Save failed.");
    } finally {
      setSaving(false);
    }
  }, [streamText, fieldName, sampleId, saving, saveToField,
      loadedModelId, systemPrompt, userPrompt,
      doSample, temperature, maxNewTokens, topP, topK, repPenalty, seed,
      latencyMs]); // eslint-disable-line

  // ── Derived booleans ──────────────────────────────────────────────────────
  const modelLocked = modelState === "ready" || modelState === "loading";
  const canRun      = modelState === "ready" && !!userPrompt.trim() && !!imagePath && inferenceState !== "running";
  const canSave     = !!streamText && !!fieldName.trim() && !!sampleId && !saving && inferenceState !== "running";

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div style={S.root} onKeyDown={(e) => e.stopPropagation()}>

      {/* ══════════ TOP — CONFIG ══════════ */}
      <div style={S.topSection}>

        {/* Model Setup */}
        <div>
          <div style={S.sectionLabel}>Model</div>

          <div style={{ marginBottom: 6 }}>
            <input
              style={modelLocked ? S.inputLocked : S.input}
              value={modelId}
              onChange={(e) => setModelId(e.target.value)}
              readOnly={modelLocked}
              placeholder="Any image-text-to-text model that can be run in a HF Pipeline"
            />
          </div>

          <div style={{ display: "flex", gap: 6, marginBottom: 7 }}>
            <select
              style={modelLocked ? S.selectLocked : S.select}
              value={device}
              onChange={(e) => setDevice(e.target.value)}
              disabled={modelLocked}
            >
              <option value="cuda">cuda</option>
              <option value="cpu">cpu</option>
              <option value="mps">mps</option>
            </select>
            <select
              style={modelLocked ? S.selectLocked : S.select}
              value={torchDtype}
              onChange={(e) => setTorchDtype(e.target.value)}
              disabled={modelLocked}
            >
              <option value="bfloat16">bfloat16</option>
              <option value="float16">float16</option>
              <option value="float32">float32</option>
              <option value="auto">auto</option>
            </select>
          </div>

          {modelState === "unloaded" && (
            <button
              style={modelId.trim() ? S.btnLoad : S.btnSmallDisabled}
              onClick={handleLoad}
              disabled={!modelId.trim()}
            >
              Load Model
            </button>
          )}

          {modelState === "loading" && (
            <div style={S.statusLoading}>
              <span style={S.spinner} />
              <span>Loading model…</span>
            </div>
          )}

          {modelState === "ready" && (
            <div style={S.statusReady}>
              <span style={S.statusDot} />
              <span style={S.statusText} title={loadedModelId}>{loadedModelId || "Model ready"}</span>
              {vramUsed != null && (
                <span style={S.statusVram}>
                  {vramUsed.toFixed(1)}{vramTotal != null ? `/${vramTotal.toFixed(0)}` : ""} GB
                </span>
              )}
              <button style={S.btnDanger} onClick={handleFree}>Free</button>
            </div>
          )}

          {modelState === "error" && (
            <>
              <div style={S.errorBox}>{modelError}</div>
              <button style={{ ...S.btnLoad, marginTop: 6 }} onClick={handleLoad}>Retry</button>
            </>
          )}
        </div>

        <div style={S.divider} />

        {/* System Prompt */}
        <div>
          <div style={S.sectionLabel}>
            System Prompt{" "}
            <span style={{ color: V.textDim, fontWeight: 400, textTransform: "none" as const }}>
              (optional)
            </span>
          </div>
          <textarea
            style={S.textarea}
            rows={2}
            value={systemPrompt}
            onChange={(e) => setSystemPrompt(e.target.value)}
            placeholder="e.g. You are a helpful vision assistant."
          />
        </div>

        <div style={S.divider} />

        {/* Generation Parameters — accordion */}
        <div>
          <div
            style={S.accordionHeader}
            onClick={() => setGenParamsOpen(o => !o)}
            role="button"
            aria-expanded={genParamsOpen}
          >
            <div style={S.sectionLabel}>Generation Parameters</div>
            <span style={{
              ...S.accordionChevron,
              transform: genParamsOpen ? "rotate(180deg)" : "rotate(0deg)",
            }}>▼</span>
          </div>

          {/* Smooth expand/collapse via max-height transition */}
          <div style={{
            maxHeight:  genParamsOpen ? 400 : 0,
            overflow:   "hidden",
            transition: "max-height 0.2s ease-in-out",
          }}>
            <div style={{ paddingTop: 6 }}>
              <div style={S.toggleRow}>
                <input
                  type="checkbox"
                  id="do_sample"
                  checked={doSample}
                  onChange={(e) => setDoSample(e.target.checked)}
                  style={{ width: 13, height: 13, cursor: "pointer", accentColor: V.primary }}
                />
                <label htmlFor="do_sample" style={S.toggleLabel}>do_sample</label>
              </div>

              <div style={S.paramGrid}>
                <div style={S.paramCell}>
                  <span style={doSample ? S.paramLabel : S.paramLabelDim}>Temperature</span>
                  <input type="number" style={doSample ? S.numInput : S.numInputDisabled}
                    value={temperature} onChange={(e) => setTemperature(e.target.value)}
                    disabled={!doSample} min={0} max={2} step={0.1} placeholder="0.7" />
                </div>
                <div style={S.paramCell}>
                  <span style={S.paramLabel}>Max tokens</span>
                  <input type="number" style={S.numInput}
                    value={maxNewTokens} onChange={(e) => setMaxNewTokens(e.target.value)}
                    min={1} max={4096} placeholder="512" />
                </div>
                <div style={S.paramCell}>
                  <span style={doSample ? S.paramLabel : S.paramLabelDim}>Top-p</span>
                  <input type="number" style={doSample ? S.numInput : S.numInputDisabled}
                    value={topP} onChange={(e) => setTopP(e.target.value)}
                    disabled={!doSample} min={0} max={1} step={0.05} placeholder="0.9" />
                </div>
                <div style={S.paramCell}>
                  <span style={doSample ? S.paramLabel : S.paramLabelDim}>Top-k</span>
                  <input type="number" style={doSample ? S.numInput : S.numInputDisabled}
                    value={topK} onChange={(e) => setTopK(e.target.value)}
                    disabled={!doSample} min={1} max={200} placeholder="50" />
                </div>
                <div style={S.paramCell}>
                  <span style={S.paramLabel}>Rep. penalty</span>
                  <input type="number" style={S.numInput}
                    value={repPenalty} onChange={(e) => setRepPenalty(e.target.value)}
                    min={1} max={2} step={0.1} placeholder="1.0" />
                </div>
                <div style={S.paramCell}>
                  <span style={S.paramLabel}>Seed</span>
                  <input type="number" style={S.numInput}
                    value={seed} onChange={(e) => setSeed(e.target.value)}
                    min={0} placeholder="random" />
                </div>
              </div>
            </div>
          </div>
        </div>

      </div>{/* end topSection */}

      {/* ══════════ RESIZE HANDLE ══════════ */}
      {/* Drag this bar up/down to grow or shrink the output area */}
      <div style={S.resizeHandle} onMouseDown={handleResizeMouseDown}>
        <span style={S.resizeDots}>• • •</span>
      </div>

      {/* ══════════ BOTTOM — OUTPUT ══════════ */}
      <div style={{ ...S.bottomSection, height: outputHeight }}>

        {/* Scrollable output text area */}
        <div style={S.outputArea} ref={outputScrollRef}>
          {inferenceState === "idle" && !streamText && (
            <div style={S.outputEmpty}>
              {!imagePath
                ? "Open a sample to begin."
                : modelState !== "ready"
                  ? "Load a model to begin."
                  : "Enter a prompt below and press Send."}
            </div>
          )}

          {inferenceState === "running" && !streamText && (
            <div style={S.outputLoading}>
              <span style={S.spinner} />
              <span>Starting inference…</span>
            </div>
          )}

          {streamText && (
            <div style={S.outputText}>
              <div className="vlm-md">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {streamText.replace(/\r\n/g, "\n").replace(/\r/g, "\n")}
                </ReactMarkdown>
              </div>
              {inferenceState === "running" && (
                <span style={S.cursor} />
              )}
            </div>
          )}

          {inferenceError && (
            <div style={{ ...S.errorBox, marginTop: streamText ? 8 : 0 }}>
              {inferenceError}
            </div>
          )}
        </div>

        {/* Stats footer + field name + save icon — only shown after a completed run */}
        {(inferenceState === "done" || inferenceState === "error") && streamText && (
          <div style={S.outputFooter}>
            {tokenCount != null && <span>{tokenCount} tokens</span>}
            {latencyMs  != null && <span>{(latencyMs / 1000).toFixed(2)}s</span>}
            {activeTemp != null && doSample && <span>temp={activeTemp}</span>}

            {/* Field name — combobox: free text OR pick an existing field */}
            <input
              list="vlm-field-suggestions"
              style={S.footerFieldInput}
              value={fieldName}
              onChange={(e) => { setFieldName(e.target.value); setSaveSuccess(false); setSaveError(null); }}
              onFocus={fetchFields}
              placeholder="field to save to"
            />
            {/* Datalist — only compatible ListField(DictField) fields */}
            <datalist id="vlm-field-suggestions">
              {compatibleFields.map(name => (
                <option key={name} value={name} />
              ))}
            </datalist>

            {/* Save icon button */}
            <button
              style={{
                ...S.saveIconBtn,
                cursor:  canSave ? "pointer" : "not-allowed",
                opacity: canSave ? 1 : 0.35,
                color:   saveSuccess ? V.green : V.textMuted,
              }}
              onClick={handleSave}
              disabled={!canSave}
              title={saveSuccess ? `Saved to ${fieldName}` : `Save to ${fieldName || "…"}`}
            >
              {saving ? "…" : saveSuccess ? "✓" : "💾"}
            </button>
          </div>
        )}

      </div>{/* end bottomSection */}

      {/* ══════════ PROMPT BAR — fixed at bottom ══════════ */}
      <div style={S.promptBar}>

        {/* Save error feedback (rare; shown at top of prompt bar) */}
        {saveError && <div style={S.errorBox}>{saveError}</div>}

        {/* Prompt textarea + send icon */}
        {/* Textarea wrapper — paper plane icon sits in the bottom-right corner */}
        <div style={{ position: "relative" }}>
          <textarea
            ref={promptInputRef}
            style={{
              ...(inferenceState === "running" ? S.promptInputDisabled : S.promptInput),
              paddingRight: 36,   // make room for the icon
              width: "100%",
            }}
            value={userPrompt}
            onChange={(e) => setUserPrompt(e.target.value)}
            onKeyDown={handlePromptKeyDown}
            placeholder={
              !imagePath             ? "Open a sample to begin…" :
              modelState !== "ready" ? "Load a model first…"     :
              "Type your prompt… (Enter ↵ to send, Shift+Enter for new line)"
            }
            disabled={inferenceState === "running"}
            rows={1}
          />
          {/* Paper plane send button — only visible when there's something to send */}
          <button
            onClick={handleRun}
            disabled={!canRun}
            title="Send (Enter)"
            style={{
              position:   "absolute",
              right:      8,
              bottom:     8,
              background: "none",
              border:     "none",
              padding:    0,
              cursor:     canRun ? "pointer" : "default",
              color:      inferenceState === "running" ? V.textDim
                        : canRun                       ? V.primary
                        :                               V.textDim,
              opacity:    canRun || inferenceState === "running" ? 1 : 0.3,
              lineHeight: 1,
              display:    "flex",
              alignItems: "center",
            }}
          >
            {inferenceState === "running"
              /* Spinning indicator while running */
              ? <span style={{ ...S.spinner, width: 14, height: 14 }} />
              /* Paper plane SVG */
              : <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
                </svg>
            }
          </button>
        </div>

      </div>{/* end promptBar */}
    </div>
  );
};

export default VlmPromptLabPanel;
