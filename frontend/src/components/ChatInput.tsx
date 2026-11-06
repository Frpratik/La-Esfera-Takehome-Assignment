import type { FormEvent } from "react";

interface Props {
  value: string;
  disabled: boolean;
  onChange: (value: string) => void;
  onSubmit: () => void;
}

export default function ChatInput({ value, disabled, onChange, onSubmit }: Props) {
  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    if (!disabled && value.trim()) onSubmit();
  }

  return (
    <form className="chat-input" onSubmit={handleSubmit}>
      <input
        type="text"
        value={value}
        maxLength={1000} // matches the API's question length limit
        placeholder="Ask about leave, expenses, travel…"
        aria-label="Your question"
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
      />
      <button type="submit" disabled={disabled || !value.trim()}>
        Send
      </button>
    </form>
  );
}
