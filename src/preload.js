const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  uploadPhotos: (filePaths) => ipcRenderer.invoke('upload-photos', filePaths),
  getPhotos: () => ipcRenderer.invoke('get-photos'),
  showFileDialog: () => ipcRenderer.invoke('show-file-dialog'),
  getFaceClusters: () => ipcRenderer.invoke('get-face-clusters'),
  findSimilarFaces: (imagePath) => ipcRenderer.invoke('find-similar-faces', imagePath),
  deleteCluster: (clusterId) => ipcRenderer.invoke('delete-cluster', clusterId),
  deletePhoto: (photoId) => ipcRenderer.invoke('delete-photo', photoId),
  deletePhotos: (photoIds) => ipcRenderer.invoke('delete-photos', photoIds),
  onUploadProgress: (callback) => ipcRenderer.on('upload-progress', callback),
  onDeleteProgress: (callback) => ipcRenderer.on('delete-progress', callback),
  removeAllListeners: (channel) => ipcRenderer.removeAllListeners(channel)
});