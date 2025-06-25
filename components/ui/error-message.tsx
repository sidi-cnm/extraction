// Reusable error message component
// Displays user-friendly error messages throughout the app

import { AlertCircle } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ErrorMessageProps {
  message: string;
  className?: string;
  variant?: 'default' | 'destructive';
}

export function ErrorMessage({ message, className, variant = 'destructive' }: ErrorMessageProps) {
  return (
    <div className={cn(
      'flex items-center space-x-2 p-4 rounded-lg border',
      variant === 'destructive' 
        ? 'bg-red-50 border-red-200 text-red-800' 
        : 'bg-gray-50 border-gray-200 text-gray-800',
      className
    )}>
      <AlertCircle className="h-5 w-5 flex-shrink-0" />
      <span className="text-sm">{message}</span>
    </div>
  );
}