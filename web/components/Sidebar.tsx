import { FolderGit2, Settings, Sparkles, X } from 'lucide-react';

interface SidebarProps {
  workingDir: string;
  setWorkingDir: (dir: string) => void;
  onClose?: () => void;
  showCloseButton?: boolean;
}

export default function Sidebar({
  workingDir,
  setWorkingDir,
  onClose,
  showCloseButton = false,
}: SidebarProps) {
  return (
    <aside className="flex h-full w-72 flex-col justify-between overflow-y-auto border-r border-white/5 bg-[#09090B] p-5 shadow-2xl sm:p-6">
      <div>
        {/* Logo Section */}
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

        {/* Directory Input Section */}
        <div className="space-y-4">
          <label className="flex items-center gap-2 px-2 text-[10px] font-bold uppercase tracking-widest text-gray-500">
            <FolderGit2 className="h-4 w-4" />
            Workspace
          </label>

          <div className="group relative">
            {/* Glowing border effect on hover */}
            <div className="absolute -inset-0.5 rounded-xl bg-gradient-to-r from-indigo-500 to-cyan-400 opacity-0 blur transition duration-500 group-hover:opacity-20" />

            <input
              type="text"
              placeholder="e.g. /Users/dev/project"
              value={workingDir}
              onChange={(e) => setWorkingDir(e.target.value)}
              className="relative w-full rounded-xl border border-white/10 bg-[#18181B] p-3.5 text-sm text-gray-200 shadow-inner placeholder-gray-600 transition-all focus:border-indigo-500/50 focus:outline-none focus:ring-1 focus:ring-indigo-500/50"
            />
          </div>

          <p className="px-2 text-[11px] leading-relaxed text-gray-500">
            Link a local directory to enable autonomous file editing and
            terminal commands. Leave blank for general chat.
          </p>
        </div>
      </div>

      {/* Bottom Settings Link */}
      <button
        type="button"
        className="flex w-full items-center gap-3 rounded-lg px-2 py-3 text-left text-gray-500 transition hover:bg-white/5 hover:text-gray-300"
      >
        <Settings className="h-4 w-4" />

        <span className="text-sm font-medium">
          Agent Settings
        </span>
      </button>
    </aside>
  );
}