const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const http = require('http');

let apiProcess = null;

function checkBackendRunning() {
  return new Promise((resolve) => {
    const req = http.get('http://127.0.0.1:8000/docs', (res) => {
      resolve(res.statusCode < 500);
    });
    req.on('error', () => resolve(false));
    req.setTimeout(500, () => {
      req.destroy();
      resolve(false);
    });
  });
}

async function ensureBackend() {
  const isRunning = await checkBackendRunning();
  if (!isRunning) {
    const rootDir = path.resolve(__dirname, '..');
    const venvPython = path.join(rootDir, 'venv', 'bin', 'python');
    apiProcess = spawn(venvPython, ['-m', 'uvicorn', 'api:app', '--host', '127.0.0.1', '--port', '8000'], {
      cwd: path.join(rootDir, 'api'),
      stdio: 'ignore',
    });
  }
}

async function createWindow() {
  await ensureBackend();

  const win = new BrowserWindow({
    width: 1200,
    height: 800,
    minWidth: 800,
    minHeight: 600,
    titleBarStyle: 'hiddenInset', // Clean, native macOS look
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false, // Required to allow Next.js to use IPC directly
    },
  });

  // Load the Next.js local server
  win.loadURL('http://localhost:3000');
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('quit', () => {
  if (apiProcess) {
    try {
      apiProcess.kill();
    } catch (e) {}
  }
});

// This listens for the frontend asking to open the native folder picker
ipcMain.handle('dialog:openDirectory', async () => {
  const result = await dialog.showOpenDialog({
    properties: ['openDirectory'], // Forces it to only select folders
  });
  
  if (result.canceled) {
    return null;
  } else {
    return result.filePaths[0]; // Returns the selected absolute path
  }
});