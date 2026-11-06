import { useRef, useState } from "react";
import { ApiError, askQuestion, sendFeedback } from "./api";
import ChatInput from "./components/ChatInput";
import MessageList from "./components/MessageList";
import type { ChatMessage, Rating } from "./types";
import "./styles.css";

function messageOf(err: unknown): string {
  return err instanceof ApiError ? err.message : "Something went wrong. Please try again.";
}

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const nextId = useRef(1);

  async function send() {
    const question = draft.trim();
    if (!question || loading) return;

    const userId = nextId.current++;
    setMessages((prev) => [...prev, { id: userId, role: "user", text: question }]);
    setDraft("");
    setError(null);
    setLoading(true);
    try {
      const res = await askQuestion(question);
      setMessages((prev) => [
        ...prev,
        {
          id: nextId.current++,
          role: "assistant",
          text: res.answer,
          sources: res.sources,
          queryId: res.query_id,
          latencyMs: res.latency_ms,
          rating: null,
        },
      ]);
    } catch (err) {
      // Undo the optimistic user message and restore the text so the user can just resend.
      setMessages((prev) => prev.filter((m) => m.id !== userId));
      setDraft(question);
      setError(messageOf(err));
    } finally {
      setLoading(false);
    }
  }

  async function rate(messageId: number, rating: Rating) {
    const target = messages.find((m) => m.id === messageId);
    if (!target || target.role !== "assistant" || target.queryId === null) return;

    const previous = target.rating;
    const update = (value: Rating | null) =>
      setMessages((prev) =>
        prev.map((m) => (m.id === messageId && m.role === "assistant" ? { ...m, rating: value } : m)),
      );
    update(rating); // optimistic: the button lights up immediately
    try {
      await sendFeedback(target.queryId, rating);
    } catch (err) {
      update(previous);
      setError(messageOf(err));
    }
  }

  return (
    <div className="app">
      <header>
        <h1>Northwind Facilities Helpdesk</h1>
        <p>Answers come only from Northwind Facilities policy documents.</p>
      </header>
      <MessageList messages={messages} loading={loading} onRate={rate} onPickExample={setDraft} />
      {error && (
        <div className="error" role="alert">
          <span>{error}</span>
          <button type="button" aria-label="Dismiss error" onClick={() => setError(null)}>
            ×
          </button>
        </div>
      )}
      <ChatInput value={draft} disabled={loading} onChange={setDraft} onSubmit={send} />
    </div>
  );
}
