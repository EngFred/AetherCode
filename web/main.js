const { app, BrowserWindow, ipcMain, dialog } = require('electron');

function createWindow() {
  const win = new BrowserWindow({
    width: 1200,
    height: 800,
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