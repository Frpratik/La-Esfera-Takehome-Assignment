import type { Rating } from "../types";

interface Props {
  rating: Rating | null;
  onRate: (rating: Rating) => void;
}

/** Thumbs up/down; the chosen one is highlighted and exposed via aria-pressed. */
export default function FeedbackButtons({ rating, onRate }: Props) {
  return (
    <div className="feedback" role="group" aria-label="Rate this answer">
      <button
        type="button"
        aria-label="Helpful"
        aria-pressed={rating === "up"}
        className={rating === "up" ? "active" : ""}
        onClick={() => onRate("up")}
      >
        👍
      </button>
      <button
        type="button"
        aria-label="Not helpful"
        aria-pressed={rating === "down"}
        className={rating === "down" ? "active" : ""}
        onClick={() => onRate("down")}
      >
        👎
      </button>
    </div>
  );
}
