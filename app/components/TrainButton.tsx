'use client';

import { Loader2 } from 'lucide-react';
import { Button } from './ui/button';

interface TrainButtonProps {
  disabled?: boolean;
  loading?: boolean;
  onClick: () => void;
}

export default function TrainButton({ disabled, loading, onClick }: TrainButtonProps) {
  return (
    <Button className="w-full" disabled={disabled || loading} onClick={onClick}>
      {loading ? (
        <span className="flex items-center gap-2">
          <Loader2 className="h-4 w-4 animate-spin" />
          Starting…
        </span>
      ) : (
        'Start finetuning'
      )}
    </Button>
  );
}
