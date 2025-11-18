'use client';

import { useState } from 'react';
import { Button } from './ui/button';
import { Input } from './ui/input';

interface Props {
  endpointUrl?: string | null;
  jobId?: string | null;
}

type Message = {
  role: 'user' | 'assistant';
  text: string;
};

export default function ChatBox({ endpointUrl, jobId }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const canChat = Boolean(endpointUrl);

  function updateAssistantMessage(text: string) {
    setMessages((prev) => {
      if (prev.length === 0) {
        return prev;
      }
      const clone = [...prev];
      const lastIndex = clone.length - 1;
      if (clone[lastIndex]?.role === 'assistant') {
        clone[lastIndex] = { role: 'assistant', text };
      }
      return clone;
    });
  }

  async function sendMessage(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!input.trim() || !canChat || pending) {
      return;
    }

    const newMessage: Message = { role: 'user', text: input };
    setMessages((prev) => [...prev, newMessage, { role: 'assistant', text: '' }]);
    setInput('');
    setPending(true);
    setError(null);

    try {
      const res = await fetch('/api/inference', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ endpoint_url: endpointUrl, job_id: jobId, message: newMessage.text, stream: true })
      });

      if (!res.ok) {
        const message = await res.text();
        throw new Error(message || 'Inference failed. Please try again.');
      }

      if (res.body) {
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let aggregated = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) {
            break;
          }
          aggregated += decoder.decode(value, { stream: true });
          updateAssistantMessage(aggregated);
        }
        updateAssistantMessage(aggregated.trim() || 'No content returned');
      } else {
        const data = await res.json();
        updateAssistantMessage(data.message ?? 'No content returned');
      }
    } catch (inferenceError) {
      const message = inferenceError instanceof Error ? inferenceError.message : 'Inference failed. Please try again.';
      updateAssistantMessage('Unable to generate response.');
      setError(message);
    } finally {
      setPending(false);
    }
  }

  return (
    <div className="flex h-full flex-col rounded-2xl border border-slate-900 bg-slate-950/60 p-4">
      <div className="flex-1 space-y-3 overflow-y-auto rounded-xl bg-slate-950/40 p-3">
        {messages.length === 0 ? (
          <p className="text-center text-sm text-slate-500">
            {canChat ? 'Chat with your fine-tuned model.' : 'Finishing deployment… the chat unlocks when the endpoint is ready.'}
          </p>
        ) : (
          messages.map((message, index) => (
            <div
              key={`${message.role}-${index}`}
              className={`rounded-2xl px-3 py-2 text-sm ${
                message.role === 'user' ? 'ml-auto bg-emerald-500/20 text-emerald-50' : 'mr-auto bg-slate-800/60 text-slate-100'
              }`}
            >
              {message.text}
            </div>
          ))
        )}
      </div>
      <form onSubmit={sendMessage} className="mt-4 flex items-center gap-3">
        <Input
          value={input}
          onChange={(event) => setInput(event.target.value)}
          placeholder={canChat ? 'Ask about financial filings…' : 'Deployment in progress…'}
          disabled={!canChat || pending}
        />
        <Button type="submit" disabled={!canChat || pending}>
          {pending ? '…' : 'Send'}
        </Button>
      </form>
      {error ? <p className="mt-2 text-xs text-red-400">{error}</p> : null}
    </div>
  );
}
