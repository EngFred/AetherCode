import { ArrowUp, StopCircle } from 'lucide-react';

interface ChatInputProps {
  input: string;
  setInput: (val: string) => void;
  sendMessage: () => void;
  isRunning: boolean;
}

export default function ChatInput({
  input,
  setInput,
  sendMessage,
  isRunning,
}: ChatInputProps) {
  return (
    // No longer "absolute bottom-0" — this is now a normal flow item at the
    // bottom of the flex column, so it always occupies real, reserved space
    // instead of floating on top of the message list.
    <div className="w-full bg-gradient-to-t from-[#09090B] via-[#09090B] to-transparent px-3 pb-3 pt-5 sm:px-6 sm:pb-5 sm:pt-6 lg:px-8">
      <div className="mx-auto w-full max-w-3xl">
        <div className="group relative">
          {/* Animated glowing backdrop when typing */}
          <div className="absolute -inset-0.5 rounded-3xl bg-gradient-to-r from-indigo-500 to-cyan-400 opacity-10 blur transition duration-500 group-focus-within:opacity-30" />

          <div className="relative flex items-end gap-2 rounded-3xl border border-white/10 bg-[#18181B] px-3 py-2.5 shadow-2xl sm:gap-3 sm:pl-5 sm:pr-3 sm:py-3">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  sendMessage();
                }
              }}
              placeholder="Ask Aether to build or modify something..."
              disabled={isRunning}
              rows={1}
              className="min-h-[24px] max-h-40 flex-1 resize-none overflow-y-auto bg-transparent py-2 text-sm leading-relaxed text-gray-100 placeholder-gray-600 focus:outline-none disabled:opacity-50 sm:text-base"
            />

            <button
              onClick={sendMessage}
              disabled={!input.trim() && !isRunning}
              aria-label={isRunning ? 'Stop generation' : 'Send message'}
              className={`flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-full transition-all duration-300 sm:h-11 sm:w-11 ${
                isRunning
                  ? 'bg-red-500/10 text-red-400 hover:bg-red-500/20'
                  : input.trim()
                    ? 'bg-white text-black shadow-md hover:scale-105 hover:bg-gray-200'
                    : 'bg-white/5 text-gray-600'
              }`}
            >
              {isRunning ? (
                <StopCircle className="h-5 w-5 animate-pulse" />
              ) : (
                <ArrowUp className="h-5 w-5" />
              )}
            </button>
          </div>
        </div>

        <p className="hidden px-2 pt-2 text-center text-[10px] text-gray-600 sm:block">
          Press Enter to send · Shift + Enter for a new line
        </p>
      </div>
    </div>
  );
}