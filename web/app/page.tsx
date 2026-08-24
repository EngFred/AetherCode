'use client';

import React, { useState, useRef, useEffect } from 'react';
import { Command, Cpu, Menu, Sparkles, User, X } from 'lucide-react';
import Sidebar from '@/components/Sidebar';
import ChatInput from '@/components/ChatInput';
import MarkdownMessage from '@/components/MarkdownMessage';

type Message = {
  role: 'user' | 'ai' | 'system';
  text: string;
};

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [workingDir, setWorkingDir] = useState('');
  const [isRunning, setIsRunning] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({
      behavior: 'smooth',
      block: 'end',
    });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = () => {
    if (!input.trim() || isRunning) return;

    const ws = new WebSocket('ws://localhost:8000/ws/chat');

    setIsRunning(true);
    setMessages((prev) => [
      ...prev,
      {
        role: 'user',
        text: input,
      },
    ]);

    ws.onopen = () => {
      ws.send(
        JSON.stringify({
          prompt: input,
          working_dir: workingDir,
        }),
      );

      setInput('');
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.text === '--- END OF TASK ---') {
        setIsRunning(false);
        ws.close();
      } else {
        setMessages((prev) => [
          ...prev,
          {
            role: data.role,
            text: data.text,
          },
        ]);
      }
    };

    ws.onerror = () => {
      setMessages((prev) => [
        ...prev,
        {
          role: 'system',
          text: '❌ Connection to API failed.',
        },
      ]);

      setIsRunning(false);
    };
  };

  return (
    <div className="flex h-[100dvh] overflow-hidden bg-[#09090B] font-sans text-gray-100 selection:bg-indigo-500/30">
      {/* Desktop Sidebar */}
      <div className="hidden lg:block h-full flex-shrink-0">
        <Sidebar
          workingDir={workingDir}
          setWorkingDir={setWorkingDir}
        />
      </div>

      {/* Mobile / Tablet Sidebar Overlay */}
      <div
        className={`fixed inset-0 z-50 lg:hidden ${
          isSidebarOpen ? 'pointer-events-auto' : 'pointer-events-none'
        }`}
      >
        {/* Backdrop */}
        <button
          type="button"
          aria-label="Close sidebar"
          onClick={() => setIsSidebarOpen(false)}
          className={`absolute inset-0 bg-black/60 backdrop-blur-sm transition-opacity duration-300 ${
            isSidebarOpen ? 'opacity-100' : 'opacity-0'
          }`}
        />

        {/* Drawer */}
        <div
          className={`absolute left-0 top-0 h-full w-[min(18rem,85vw)] transform transition-transform duration-300 ease-out ${
            isSidebarOpen ? 'translate-x-0' : '-translate-x-full'
          }`}
        >
          <Sidebar
            workingDir={workingDir}
            setWorkingDir={setWorkingDir}
            onClose={() => setIsSidebarOpen(false)}
            showCloseButton
          />
        </div>
      </div>

      {/* Main Application Area */}
      <main className="relative flex min-w-0 flex-1 flex-col">
        {/* Subtle radial gradient background */}
        <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-indigo-900/10 via-[#09090B] to-[#09090B]" />

        {/* Mobile Header */}
        <header className="relative z-20 flex h-16 flex-shrink-0 items-center justify-between border-b border-white/5 bg-[#09090B]/80 px-4 backdrop-blur-xl lg:hidden">
          <button
            type="button"
            onClick={() => setIsSidebarOpen(true)}
            className="flex h-10 w-10 items-center justify-center rounded-xl border border-white/10 bg-white/[0.03] text-gray-300 transition hover:bg-white/[0.07] hover:text-white"
            aria-label="Open sidebar"
          >
            <Menu className="h-5 w-5" />
          </button>

          <div className="flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-indigo-500 to-cyan-400 shadow-[0_0_15px_rgba(99,102,241,0.3)]">
              <Sparkles className="h-4 w-4 text-white" />
            </div>

            <span className="text-sm font-semibold tracking-tight text-gray-100">
              AetherCode
            </span>
          </div>

          {/* Invisible spacer to keep logo centered */}
          <div className="h-10 w-10" />
        </header>

        {/* min-h-0 and min-w-0 are important here.
            They allow this flex child to shrink correctly and prevent
            long message content from forcing horizontal page overflow. */}
        <div className="relative z-10 flex-1 min-h-0 min-w-0 overflow-y-auto overflow-x-hidden scroll-smooth">
          {messages.length === 0 ? (
            <div className="mx-auto flex h-full w-full max-w-2xl flex-col items-center justify-center px-6 py-12 text-center sm:px-8">
              <div className="group relative mb-7 cursor-default sm:mb-8">
                <div className="absolute inset-0 rounded-full bg-gradient-to-r from-indigo-500 to-cyan-400 opacity-20 blur-2xl transition duration-700 group-hover:opacity-40" />

                <div className="relative rounded-3xl border border-white/10 bg-[#18181B] p-4 shadow-2xl sm:p-5">
                  <Cpu
                    className="h-9 w-9 text-indigo-400 sm:h-10 sm:w-10"
                    strokeWidth={1.5}
                  />
                </div>
              </div>

              <h2 className="mb-3 bg-gradient-to-b from-white to-gray-500 bg-clip-text text-2xl font-semibold tracking-tight text-transparent sm:text-3xl">
                How can I help you code?
              </h2>

              <p className="max-w-lg text-sm leading-relaxed text-gray-500 sm:text-base">
                I am Aether, your autonomous local AI. Point me to a project
                directory, or just ask me a general programming question to
                begin.
              </p>
            </div>
          ) : (
            <div className="mx-auto w-full max-w-4xl px-4 py-6 sm:px-6 sm:py-8 lg:px-8">
              <div className="space-y-6 sm:space-y-8">
                {messages.map((msg, idx) => (
                  <div
                    key={idx}
                    className={`flex min-w-0 gap-3 sm:gap-4 ${
                      msg.role === 'user' ? 'flex-row-reverse' : ''
                    }`}
                  >
                    {/* Avatars */}
                    <div className="mt-1 flex-shrink-0">
                      {msg.role === 'user' ? (
                        <div className="flex h-8 w-8 items-center justify-center rounded-full bg-white shadow-lg">
                          <User className="h-4 w-4 text-black sm:h-5 sm:w-5" />
                        </div>
                      ) : msg.role === 'ai' ? (
                        <div className="flex h-8 w-8 items-center justify-center rounded-full bg-gradient-to-br from-indigo-500 to-cyan-400 shadow-[0_0_10px_rgba(99,102,241,0.4)]">
                          <Sparkles className="h-4 w-4 text-white" />
                        </div>
                      ) : (
                        <div className="flex h-8 w-8 items-center justify-center rounded-full border border-white/10 bg-[#18181B]">
                          <Command className="h-4 w-4 text-gray-500" />
                        </div>
                      )}
                    </div>

                    {/* Message Bubbles */}
                    <div
                      className={`min-w-0 max-w-[calc(100%-2.75rem)] sm:max-w-[85%] ${
                        msg.role === 'user'
                          ? 'rounded-2xl border border-white/5 bg-[#18181B] px-4 py-3 text-gray-200 shadow-xl sm:px-5 sm:py-4'
                          : msg.role === 'system'
                            ? 'rounded-2xl border border-indigo-500/20 bg-[#18181B]/50 px-4 py-3 font-mono text-[11px] text-indigo-300/70 backdrop-blur-sm'
                            : 'py-1 text-gray-200'
                      }`}
                    >
                      {msg.role === 'ai' ? (
                        <MarkdownMessage content={msg.text} />
                      ) : (
                        <span className="whitespace-pre-wrap break-words leading-relaxed">
                          {msg.text}
                        </span>
                      )}
                    </div>
                  </div>
                ))}

                <div ref={messagesEndRef} />
              </div>
            </div>
          )}
        </div>

        {/* Chat Input remains in normal flex flow */}
        <div className="relative z-20 flex-shrink-0">
          <ChatInput
            input={input}
            setInput={setInput}
            sendMessage={sendMessage}
            isRunning={isRunning}
          />
        </div>
      </main>
    </div>
  );
}