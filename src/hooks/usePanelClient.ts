import { useCallback } from "react";
import { usePanelEvent } from "@fiftyone/operators";
import type { DatasetField, ModelStatus, StreamChunk } from "../types";

interface PanelUris {
  load_model: string;
  free_model: string;
  run_inference: string;
  get_model_status: string;
  get_stream_chunk: string;
  save_to_field: string;
  get_dataset_fields: string;
}

/**
 * Bridge to the Python VlmPromptLabPanel methods.
 *
 * Each method wraps ``usePanelEvent`` in a Promise so callers can use
 * async/await. Python errors (``result.result.error``) are surfaced as
 * rejected Promises.
 */
export function usePanelClient(uris: PanelUris) {
  const handleEvent = usePanelEvent();

  // Generic caller — captured via useCallback so deps stay stable
  const call = useCallback(
    <T>(methodName: string, uri: string, params: Record<string, any>): Promise<T> =>
      new Promise((resolve, reject) => {
        handleEvent(methodName, {
          operator: uri,
          params,
          callback: (result: any) => {
            const r = result?.result as (T & { error?: string }) | undefined;
            if (r?.error) {
              reject(new Error(r.error));
            } else {
              resolve(r as T);
            }
          },
        });
      }),
    [handleEvent]
  );

  const loadModel = useCallback(
    (params: { model_id: string; device: string; torch_dtype: string }) =>
      call<{ status: string }>("load_model", uris.load_model, params),
    [call, uris.load_model]
  );

  const freeModel = useCallback(
    () => call<{ status: string }>("free_model", uris.free_model, {}),
    [call, uris.free_model]
  );

  const runInference = useCallback(
    (params: {
      image_path: string;
      system_prompt: string;
      user_prompt: string;
      gen_params: Record<string, any>;
    }) => call<{ status: string }>("run_inference", uris.run_inference, params),
    [call, uris.run_inference]
  );

  const getModelStatus = useCallback(
    () => call<ModelStatus>("get_model_status", uris.get_model_status, {}),
    [call, uris.get_model_status]
  );

  const getStreamChunk = useCallback(
    (cursor: number) =>
      call<StreamChunk>("get_stream_chunk", uris.get_stream_chunk, { cursor }),
    [call, uris.get_stream_chunk]
  );

  const saveToField = useCallback(
    (params: {
      field_name: string;
      sample_id: string;
      entry: {
        model_id: string;
        system_prompt: string;
        user_prompt: string;
        generation_params: Record<string, any>;
        latency_ms: number | null;
        response: string;
      };
    }) =>
      call<{ saved: boolean; count: number }>("save_to_field", uris.save_to_field, params),
    [call, uris.save_to_field]
  );

  const getDatasetFields = useCallback(
    () => call<{ fields: DatasetField[] }>(
      "get_dataset_fields", uris.get_dataset_fields, {}
    ),
    [call, uris.get_dataset_fields]
  );

  return {
    loadModel,
    freeModel,
    runInference,
    getModelStatus,
    getStreamChunk,
    saveToField,
    getDatasetFields,
  };
}
