import { useEffect, useRef } from "react";
import type { ChatMessage, Rating } from "../types";
import MessageBubble from "./MessageBubble";

const EXAMPLES = [
  "How many days of annual leave do I get?",
  "What is the hotel limit per night?",
  "How long do I have to submit a travel claim?",
];

interface Props {
  messages: ChatMessage[];
  loading: boolean;
  onRate: (messageId: number, rating: Rating) => void;
  onPickExample: (question: string) => void;
}

export default function MessageList({ messages, loading, onRate, onPickExample }: Props) {
  const endRef = useRef<HTMLDivElement>(null);

  // Keep the newest message (or the typing indicator) in view.
  useEffect(() => {
    endRef.current?.scrollIntoView?.({ behavior: "smooth", block: "end" });
  }, [messages, loading]);

  return (
    <div className="messages" role="log" aria-live="polite" aria-label="Conversation">
      {messages.length === 0 && !loading && (
        <div className="empty">
          <p>Ask a question about company policies.</p>
          <div className="examples">
            {EXAMPLES.map((q) => (
              <button key={q} type="button" onClick={() => onPickExample(q)}>
                {q}
              </button>
            ))}
          </div>
        </div>
      )}
      {messages.map((m) => (
        <MessageBubble key={m.id} message={m} onRate={onRate} />
      ))}
      {loading && (
        <div className="message assistant" aria-label="Assistant is typing">
          <div className="bubble typing">
            <span />
            <span />
            <span />
          </div>
        </div>
      )}
      <div ref={endRef} />
    </div>
  );
}
