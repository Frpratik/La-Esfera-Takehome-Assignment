import type { ChatMessage, Rating } from "../types";
import FeedbackButtons from "./FeedbackButtons";
import Sources from "./Sources";

const REFUSAL = "I don't know";

interface Props {
  message: ChatMessage;
  onRate: (messageId: number, rating: Rating) => void;
}

export default function MessageBubble({ message, onRate }: Props) {
  if (message.role === "user") {
    return (
      <div className="message user">
        <div className="bubble">{message.text}</div>
      </div>
    );
  }

  // A refusal has no real supporting sources, so don't present retrieval noise as citations.
  const refused = message.text.trim() === REFUSAL;
  return (
    <div className="message assistant">
      <div className="bubble">{message.text}</div>
      {!refused && <Sources sources={message.sources} />}
      <div className="meta">
        <span>{Math.round(message.latencyMs)} ms</span>
        {message.queryId !== null && (
          <FeedbackButtons rating={message.rating} onRate={(r) => onRate(message.id, r)} />
        )}
      </div>
    </div>
  );
}
