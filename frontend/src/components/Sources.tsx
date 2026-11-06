import type { Source } from "../types";

/** Citations shown under an answer: which policy and chunk, and how close the match was. */
export default function Sources({ sources }: { sources: Source[] }) {
  if (sources.length === 0) return null;
  return (
    <div className="sources">
      <span className="sources-label">Sources</span>
      <ul>
        {sources.map((s) => (
          <li key={s.chunk_id} title={`similarity ${s.score.toFixed(3)}`}>
            <span className="source-doc">{s.doc.replace(/_/g, " ")}</span>
            <span className="source-id">{s.chunk_id}</span>
            <span className="source-score">{s.score.toFixed(2)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
