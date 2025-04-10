const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const fs = require('fs-extra');
const { parse } = require('csv-parse/sync');
const NeuralNetworkManager = require('./neuralNetwork');

let mainWindow;
let neuralNetwork;

// Handle creating/removing shortcuts on Windows when installing/uninstalling.
if (require('electron-squirrel-startup')) {
  app.quit();
}

function createWindow() {
  // Create the browser window.
  mainWindow = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false
    }
  });

  // and load the index.html of the app.
  mainWindow.loadFile(path.join(__dirname, 'index.html'));

  // Open the DevTools.
  mainWindow.webContents.openDevTools();
}

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.whenReady().then(() => {
  createWindow();
  neuralNetwork = new NeuralNetworkManager();

  // On OS X it's common to re-create a window in the app when the
  // dock icon is clicked and there are no other windows open.
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

// Quit when all windows are closed, except on macOS. There, it's common
// for applications and their menu bar to stay active until the user quits
// explicitly with Cmd + Q.
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// In this file you can include the rest of your app's specific main process
// code. You can also put them in separate files and import them here.

// Обработчик для получения последних значений и предсказания
ipcMain.handle('get-prediction', async () => {
    try {
        const data = await fs.readFile(path.join(__dirname, 'data.csv'), 'utf-8');
        const records = parse(data, {
            columns: true,
            skip_empty_lines: true
        });
        
        const values = records.map(record => parseFloat(record.value)).filter(v => !isNaN(v));
        const lastValues = values.slice(-15);
        
        // Обучаем только если есть новые данные
        if (values.length > neuralNetwork.lastDataLength) {
            await neuralNetwork.train(values);
        }
        
        const predictions = [];
        if (lastValues.length > 0) {
            for (let i = 0; i < lastValues.length; i++) {
                const prediction = neuralNetwork.predict(lastValues.slice(0, i + 1));
                predictions.push(prediction);
            }
        }
        
        const nextPrediction = neuralNetwork.predict(lastValues);
        
        return {
            lastValues,
            predictions,
            prediction: nextPrediction
        };
    } catch (error) {
        console.error('Error:', error);
        return { 
            lastValues: [],
            predictions: [],
            prediction: 0,
            error: error.message 
        };
    }
});
