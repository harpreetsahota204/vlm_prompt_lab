export interface ModelStatus {
  status: "unloaded" | "loading_model" | "ready" | "load_error";
  model_id?: string;
  vram_gb?: number;
  total_vram_gb?: number;
  message?: string;
  error?: string;
}

export interface StreamChunk {
  text: string;
  cursor: number;
  done: boolean;
  final_status?: {
    status: "done" | "inference_error";
    token_count?: number;
    latency_ms?: number;
    temperature?: number;
    error?: string;
  };
}

export interface PanelData {
  sample_id?: string;
  image_path?: string;
  initial_model_status?: ModelStatus;
}

export interface DatasetField {
  name: string;
  kind: "list_dict";
}

export interface PanelSchema {
  view?: {
    load_model?: string;
    free_model?: string;
    run_inference?: string;
    get_model_status?: string;
    get_stream_chunk?: string;
    save_to_field?: string;
    get_dataset_fields?: string;
  };
}
