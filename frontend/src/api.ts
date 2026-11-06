import type { AskResponse, Rating } from "./types";

const BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000").replace(/\/$/, "");
const API_KEY = import.meta.env.VITE_API_KEY ?? "";
const TIMEOUT_MS = 30_000;

/** An error whose message is safe to show directly to the user. */
export class ApiError extends Error {}

async function request<T>(path: string, body: unknown): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  let response: Response;
  try {
    response = await fetch(`${BASE_URL}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-API-Key": API_KEY },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new ApiError("The request timed out. Please try again.");
    }
    throw new ApiError("Could not reach the server. Is the API running?");
  } finally {
    clearTimeout(timer);
  }

  if (!response.ok) {
    if (response.status === 401) {
      throw new ApiError("Not authorised: check VITE_API_KEY matches the server's API_KEY.");
    }
    if (response.status === 503) {
      throw new ApiError("The knowledge index is not ready yet. Try again shortly.");
    }
    throw new ApiError(`The server returned an error (${response.status}).`);
  }
  return (await response.json()) as T;
}

export function askQuestion(question: string): Promise<AskResponse> {
  return request<AskResponse>("/ask", { question });
}

export async function sendFeedback(queryId: number, rating: Rating): Promise<void> {
  await request("/feedback", { query_id: queryId, rating });
}
