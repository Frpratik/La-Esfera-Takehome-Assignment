/** Shapes returned by the API (mirror app/schemas.py). */

export interface Source {
  doc: string;
  chunk_id: string;
  score: number;
}

export interface AskResponse {
  query_id: number | null;
  answer: string;
  sources: Source[];
  latency_ms: number;
}

export type Rating = "up" | "down";

/** A single entry in the chat transcript. */
export type ChatMessage =
  | { id: number; role: "user"; text: string }
  | {
      id: number;
      role: "assistant";
      text: string;
      sources: Source[];
      queryId: number | null;
      latencyMs: number;
      rating: Rating | null;
    };
