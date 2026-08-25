import { FolderGit2, Gauge, MessageSquarePlus, Settings, Sparkles, Telescope, X, Zap, FolderSearch } from 'lucide-react';

type ExecutionMode = 'direct' | 'auto' | 'deep';

interface SidebarProps {
  workingDir: string;
  setWorkingDir: (dir: string) => void;
  executionMode: ExecutionMode;
  setExecutionMode: (mode: ExecutionMode) => void;
  onNewChat: () => void;
  onClose?: () => void;
  showCloseButton?: boolean;
}

const MODES: { id: ExecutionMode; label: string; description: string; icon: React.ElementType }[] = [
  { id: 'direct', label: 'Instant', description: 'Groq only. Fastest & cheapest — use when you already know the file(s).', icon: Zap },
  { id: 'auto', label: 'Auto', description: 'Smart routing: skips the scan when your prompt names a file; runs full analysis otherwise.', icon: Gauge },
  { id: 'deep', label: 'Deep Scan', description: 'Always scans the full project with Gemini first. Best for broad or unfamiliar changes.', icon: Telescope },
];

export default function Sidebar({
  workingDir,
  setWorkingDir,
  executionMode,
  setExecutionMode,
  onNewChat,
  onClose,
  showCloseButton = false,
}: SidebarProps) {
  
  const handleNewChat = () => {
    onNewChat();
    if (showCloseButton) onClose?.();
  };

  // Triggers the native macOS folder picker via Electron IPC
  const handleBrowse = async () => {
    if (typeof window !== 'undefined' && window.require) {
      const { ipcRenderer } = window.require('electron');
      const selectedPath = await ipcRenderer.invoke('dialog:openDirectory');
      if (selectedPath) {
        setWorkingDir(selectedPath);
      }
    } else {
      alert("Native folder picking is only available in the Electron desktop app.");
    }
  };

  return (
    <aside className="flex h-full w-72 flex-col justify-between overflow-y-auto border-r border-white/5 bg-[#09090B] p-5 shadow-2xl sm:p-6">
      <div>
        {/* Header */}
        <div className="mb-8 flex items-center justify-between gap-3 px-2 sm:mb-10">
          <div className="flex items-center gap-3">
            <div className="rounded-xl bg-gradient-to-br from-indigo-500 to-cyan-400 p-2 shadow-[0_0_15px_rgba(99,102,241,0.4)]">
              <Sparkles className="h-5 w-5 text-white" />
            </div>
            <h1 className="bg-gradient-to-r from-white to-gray-400 bg-clip-text text-xl font-bold tracking-tight text-transparent">
              AetherCode
            </h1>
          </div>
          {showCloseButton && (
            <button
              type="button"
              onClick={onClose}
              aria-label="Close sidebar"
              className="flex h-9 w-9 items-center justify-center rounded-lg text-gray-500 transition hover:bg-white/5 hover:text-white"
            >
              <X className="h-5 w-5" />
            </button>
          )}
        </div>

        {/* New Chat Button */}
        <button
          type="button"
          onClick={handleNewChat}
          className="mb-8 flex w-full items-center justify-center gap-2 rounded-xl border border-white/10 bg-[#18181B] px-3.5 py-3 text-sm font-medium text-gray-200 shadow-inner transition-all hover:border-indigo-500/40 hover:bg-white/[0.06] hover:text-white sm:mb-10"
        >
          <MessageSquarePlus className="h-4 w-4" />
          New Chat
        </button>

        {/* Workspace Input with Browse Button */}
        <div className="space-y-4">
          <label className="flex items-center gap-2 px-2 text-[10px] font-bold uppercase tracking-widest text-gray-500">
            <FolderGit2 className="h-4 w-4" />
            Workspace
          </label>

          <div className="group relative">
            <div className="absolute -inset-0.5 rounded-xl bg-gradient-to-r from-indigo-500 to-cyan-400 opacity-0 blur transition duration-500 group-hover:opacity-20" />
            
            <div className="relative flex items-center w-full rounded-xl border border-white/10 bg-[#18181B] shadow-inner focus-within:border-indigo-500/50 focus-within:ring-1 focus-within:ring-indigo-500/50 transition-all">
              <input
                type="text"
                placeholder="e.g. /Users/dev/project"
                value={workingDir}
                onChange={(e) => setWorkingDir(e.target.value)}
                className="w-full bg-transparent p-3.5 text-sm text-gray-200 placeholder-gray-600 focus:outline-none"
              />
              <button 
                onClick={handleBrowse}
                title="Browse Folders"
                className="mr-2 flex items-center justify-center rounded-lg p-2 text-gray-400 hover:bg-white/10 hover:text-white transition-colors"
              >
                <FolderSearch className="h-4 w-4" />
              </button>
            </div>
          </div>

          <p className="px-2 text-[11px] leading-relaxed text-gray-500">
            Link a local directory to enable autonomous file editing and terminal commands. Leave blank for general chat.
          </p>
        </div>

        {/* Modes Section */}
        <div className="mt-8 space-y-3">
          <label className="flex items-center gap-2 px-2 text-[10px] font-bold uppercase tracking-widest text-gray-500">
            <Gauge className="h-4 w-4" />
            Mode
          </label>

          <div className="space-y-2">
            {MODES.map(({ id, label, description, icon: Icon }) => (
              <button
                key={id}
                type="button"
                onClick={() => setExecutionMode(id)}
                className={`w-full rounded-xl border p-3 text-left transition-all ${
                  executionMode === id
                    ? 'border-indigo-500/50 bg-indigo-500/10 shadow-[0_0_15px_rgba(99,102,241,0.15)]'
                    : 'border-white/10 bg-[#18181B] hover:border-white/20'
                }`}
              >
                <div className="flex items-center gap-2">
                  <Icon className={`h-4 w-4 ${executionMode === id ? 'text-indigo-400' : 'text-gray-500'}`} />
                  <span className={`text-sm font-medium ${executionMode === id ? 'text-white' : 'text-gray-300'}`}>
                    {label}
                  </span>
                </div>
                <p className="mt-1 text-[11px] leading-relaxed text-gray-500">{description}</p>
              </button>
            ))}
          </div>

          {!workingDir && (
            <p className="px-2 text-[10px] leading-relaxed text-gray-600">
              No workspace linked — every mode runs Groq only until you set one.
            </p>
          )}
        </div>
      </div>

      <button
        type="button"
        className="flex w-full items-center gap-3 rounded-lg px-2 py-3 text-left text-gray-500 transition hover:bg-white/5 hover:text-gray-300"
      >
        <Settings className="h-4 w-4" />
        <span className="text-sm font-medium">Agent Settings</span>
      </button>
    </aside>
  );
}