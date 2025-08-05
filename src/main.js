const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const fs = require('fs').promises;
const exifr = require('exifr');
const sharp = require('sharp');
const axios = require('axios');
const FormData = require('form-data');
const sqlite3 = require('sqlite3').verbose();
const { spawn } = require('child_process');

let mainWindow;
let pythonProcess;

const PHOTOS_DIR = path.join(__dirname, '../photos');
const THUMBNAILS_DIR = path.join(__dirname, '../thumbnails');
let FACE_SERVICE_URL = 'http://127.0.0.1:8000'; // Will be updated dynamically
const DB_FILE = path.join(__dirname, '../photos.db');

let db;

async function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js')
    }
  });

  mainWindow.loadFile(path.join(__dirname, 'index.html'));

  if (process.argv.includes('--dev')) {
    mainWindow.webContents.openDevTools();
  }
}

async function initializeDirectories() {
  try {
    await fs.mkdir(PHOTOS_DIR, { recursive: true });
    await fs.mkdir(THUMBNAILS_DIR, { recursive: true });
  } catch (error) {
    console.error('Error creating directories:', error);
  }
}

function initializeDatabase() {
  return new Promise((resolve, reject) => {
    db = new sqlite3.Database(DB_FILE, (err) => {
      if (err) {
        console.error('Error opening database:', err);
        reject(err);
        return;
      }
      
      console.log('Connected to SQLite database');
      
      // Create photos table
      db.run(`CREATE TABLE IF NOT EXISTS photos (
        id TEXT PRIMARY KEY,
        filename TEXT NOT NULL,
        originalFilename TEXT,
        path TEXT NOT NULL,
        thumbnailPath TEXT,
        timestamp DATETIME,
        fileSize INTEGER,
        exif TEXT,
        faceCount INTEGER DEFAULT 0,
        createdAt DATETIME DEFAULT CURRENT_TIMESTAMP
      )`, (err) => {
        if (err) {
          console.error('Error creating photos table:', err);
          reject(err);
          return;
        }
        
        // Create faces table for normalized face storage
        db.run(`CREATE TABLE IF NOT EXISTS faces (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          photo_id TEXT NOT NULL,
          face_id INTEGER NOT NULL,
          x INTEGER NOT NULL,
          y INTEGER NOT NULL,
          width INTEGER NOT NULL,
          height INTEGER NOT NULL,
          confidence REAL DEFAULT 0.0,
          face_image TEXT,
          embedding TEXT,
          cluster_id INTEGER,
          createdAt DATETIME DEFAULT CURRENT_TIMESTAMP,
          FOREIGN KEY (photo_id) REFERENCES photos (id) ON DELETE CASCADE
        )`, (err) => {
          if (err) {
            console.error('Error creating faces table:', err);
            reject(err);
            return;
          }
          
          // Create face_clusters table for grouping faces
          db.run(`CREATE TABLE IF NOT EXISTS face_clusters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            face_count INTEGER DEFAULT 0,
            representative_face_id INTEGER,
            similarity_threshold REAL DEFAULT 0.6,
            createdAt DATETIME DEFAULT CURRENT_TIMESTAMP,
            updatedAt DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (representative_face_id) REFERENCES faces (id)
          )`, (err) => {
            if (err) {
              console.error('Error creating face_clusters table:', err);
              reject(err);
              return;
            }
            
            // Add originalFilename column if it doesn't exist (migration)
            db.run('ALTER TABLE photos ADD COLUMN originalFilename TEXT', (err) => {
              // Ignore error if column already exists
              if (err && !err.message.includes('duplicate column name')) {
                console.error('Error adding originalFilename column:', err);
              }
              
              // Update existing photos to have originalFilename = filename
              db.run('UPDATE photos SET originalFilename = filename WHERE originalFilename IS NULL', (err) => {
                if (err) {
                  console.error('Error updating originalFilename:', err);
                }
                
                // Create indexes for performance
                const indexes = [
                  'CREATE INDEX IF NOT EXISTS idx_faces_photo_id ON faces(photo_id)',
                  'CREATE INDEX IF NOT EXISTS idx_faces_cluster_id ON faces(cluster_id)',
                  'CREATE INDEX IF NOT EXISTS idx_photos_timestamp ON photos(timestamp)',
                  'CREATE INDEX IF NOT EXISTS idx_photos_filename ON photos(filename)'
                ];
                
                let indexCount = 0;
                indexes.forEach(indexSql => {
                  db.run(indexSql, (err) => {
                    indexCount++;
                    if (err) {
                      console.error('Error creating index:', err);
                    }
                    if (indexCount === indexes.length) {
                      console.log('Database tables and indexes ready');
                      resolve();
                    }
                  });
                });
              });
            });
          });
        });
      });
    });
  });
}

function getAllPhotos() {
  return new Promise((resolve, reject) => {
    db.all(`SELECT p.*, 
            COUNT(f.id) as actualFaceCount,
            GROUP_CONCAT(f.id) as faceIds
            FROM photos p 
            LEFT JOIN faces f ON p.id = f.photo_id 
            GROUP BY p.id 
            ORDER BY p.timestamp DESC`, (err, rows) => {
      if (err) {
        console.error('Error loading photos from DB:', err);
        reject(err);
      } else {
        // Parse JSON fields and include face information
        const photos = rows.map(row => ({
          ...row,
          exif: row.exif ? JSON.parse(row.exif) : {},
          faceCount: row.actualFaceCount || 0,
          faceIds: row.faceIds ? row.faceIds.split(',').map(id => parseInt(id)) : []
        }));
        resolve(photos);
      }
    });
  });
}

function addPhotoToDB(photo) {
  return new Promise((resolve, reject) => {
    // Use provided ID or generate new one
    if (!photo.id) {
      photo.id = Date.now().toString() + Math.random().toString(36).substr(2, 9);
    }
    
    const stmt = db.prepare(`INSERT INTO photos 
      (id, filename, originalFilename, path, thumbnailPath, timestamp, fileSize, exif, createdAt) 
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`);
    
    stmt.run([
      photo.id,
      photo.filename,
      photo.originalFilename || photo.filename,
      photo.path,
      photo.thumbnailPath,
      photo.timestamp,
      photo.fileSize,
      JSON.stringify(photo.exif || {}),
      new Date().toISOString()
    ], function(err) {
      if (err) {
        console.error('Error adding photo to DB:', err);
        reject(err);
      } else {
        resolve(photo);
      }
    });
    
    stmt.finalize();
  });
}

function addFacesToDB(faces, photoId) {
  return new Promise((resolve, reject) => {
    if (!faces || faces.length === 0) {
      resolve([]);
      return;
    }
    
    const stmt = db.prepare(`INSERT INTO faces 
      (photo_id, face_id, x, y, width, height, confidence, face_image, embedding) 
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`);
    
    const savedFaces = [];
    let processed = 0;
    
    faces.forEach((face, index) => {
      const location = face.location || {};
      stmt.run([
        photoId,
        face.face_id || index,
        location.x || 0,
        location.y || 0,
        location.width || 0,
        location.height || 0,
        face.confidence || 0.0,
        face.face_image || null,
        face.embedding ? JSON.stringify(face.embedding) : null
      ], function(err) {
        processed++;
        if (err) {
          console.error('Error adding face to DB:', err);
        } else {
          savedFaces.push({
            id: this.lastID,
            photo_id: photoId,
            face_id: face.face_id || index,
            ...location,
            confidence: face.confidence || 0.0
          });
        }
        
        if (processed === faces.length) {
          stmt.finalize();
          // Update photo face count
          updatePhotoFaceCount(photoId).then(() => {
            resolve(savedFaces);
          }).catch(reject);
        }
      });
    });
  });
}

function updatePhotoFaceCount(photoId) {
  return new Promise((resolve, reject) => {
    db.get("SELECT COUNT(*) as count FROM faces WHERE photo_id = ?", [photoId], (err, row) => {
      if (err) {
        reject(err);
      } else {
        const faceCount = row.count || 0;
        db.run("UPDATE photos SET faceCount = ? WHERE id = ?", [faceCount, photoId], (err) => {
          if (err) {
            reject(err);
          } else {
            resolve(faceCount);
          }
        });
      }
    });
  });
}

function getFacesByPhotoId(photoId) {
  return new Promise((resolve, reject) => {
    db.all("SELECT * FROM faces WHERE photo_id = ? ORDER BY face_id", [photoId], (err, rows) => {
      if (err) {
        reject(err);
      } else {
        const faces = rows.map(row => ({
          ...row,
          embedding: row.embedding ? JSON.parse(row.embedding) : null
        }));
        resolve(faces);
      }
    });
  });
}

function getAllFaces() {
  return new Promise((resolve, reject) => {
    db.all("SELECT * FROM faces ORDER BY photo_id, face_id", (err, rows) => {
      if (err) {
        reject(err);
      } else {
        const faces = rows.map(row => ({
          ...row,
          embedding: row.embedding ? JSON.parse(row.embedding) : null
        }));
        resolve(faces);
      }
    });
  });
}

function deletePhotoFromDB(photoId) {
  return new Promise((resolve, reject) => {
    // First get the photo info for file cleanup
    db.get("SELECT * FROM photos WHERE id = ?", [photoId], (err, row) => {
      if (err) {
        reject(err);
        return;
      }
      
      if (!row) {
        resolve({ success: false, message: 'Photo not found' });
        return;
      }
      
      // Delete from database
      db.run("DELETE FROM photos WHERE id = ?", [photoId], function(err) {
        if (err) {
          console.error('Error deleting photo from DB:', err);
          reject(err);
        } else {
          resolve({ success: true, photo: row });
        }
      });
    });
  });
}

async function cleanupPhotoFiles(photo) {
  const filesToDelete = [];
  
  // Add main photo file
  if (photo.path) {
    filesToDelete.push(photo.path);
  }
  
  // Add thumbnail file
  if (photo.thumbnailPath) {
    filesToDelete.push(photo.thumbnailPath);
  }
  
  // Delete files
  for (const filePath of filesToDelete) {
    try {
      await fs.unlink(filePath);
      console.log(`Deleted file: ${filePath}`);
    } catch (error) {
      console.error(`Error deleting file ${filePath}:`, error);
      // Continue with other files even if one fails
    }
  }
  
  return filesToDelete.length;
}

function startFaceDetectionService() {
  return new Promise((resolve, reject) => {
    console.log('Starting face detection service...');
    
    const serviceScript = path.join(__dirname, '../face_detection/face_service.py');
    
    // Try to find Python from the conda environment
    const condaPaths = [
      'C:\\Users\\jiaha\\anaconda3\\envs\\tensorflow\\python.exe',
      'C:\\Users\\jiaha\\anaconda3\\python.exe',
      'python'
    ];
    
    let pythonPath = 'python';
    for (const p of condaPaths) {
      try {
        require('fs').accessSync(p);
        pythonPath = p;
        break;
      } catch (e) {
        // Continue to next path
      }
    }
    
    console.log('Using Python path:', pythonPath);
    
    // Start the Python service directly
    pythonProcess = spawn(pythonPath, [serviceScript], {
      cwd: path.join(__dirname, '../face_detection'),
      stdio: ['ignore', 'pipe', 'pipe'],
      env: { ...process.env, PYTHONPATH: process.env.PYTHONPATH }
    });
    
    let serviceReady = false;
    
    pythonProcess.stdout.on('data', (data) => {
      const output = data.toString();
      console.log('Face service stdout:', output);
      
      // Check for dynamic port assignment
      const portMatch = output.match(/Using port: (\d+)/);
      if (portMatch) {
        const port = portMatch[1];
        FACE_SERVICE_URL = `http://127.0.0.1:${port}`;
        console.log(`Face service will use URL: ${FACE_SERVICE_URL}`);
      }
      
      // Check if service is ready (look for Uvicorn startup message)
      if (output.includes('Uvicorn running on') || output.includes('Application startup complete')) {
        console.log('Face detection service is ready!');
        if (!serviceReady) {
          serviceReady = true;
          resolve();
        }
      }
    });
    
    pythonProcess.stderr.on('data', (data) => {
      const error = data.toString();
      console.error('Face service stderr:', error);
      
      // Also check stderr for startup messages (some logs go to stderr)
      if (error.includes('Uvicorn running on') || error.includes('Application startup complete')) {
        console.log('Face detection service is ready (from stderr)!');
        if (!serviceReady) {
          serviceReady = true;
          resolve();
        }
      }
    });
    
    pythonProcess.on('close', (code) => {
      console.log(`Face detection service exited with code ${code}`);
      pythonProcess = null;
    });
    
    pythonProcess.on('error', (error) => {
      console.error('Failed to start face detection service:', error);
      console.log('Continuing without face detection service...');
      if (!serviceReady) {
        resolve(); // Continue anyway
      }
    });
    
    // Resolve after a longer delay if we don't see the ready message
    setTimeout(() => {
      if (!serviceReady) {
        if (pythonProcess) {
          console.log('Face detection service started (timeout fallback)');
        } else {
          console.log('Face detection service not started, continuing without it');
        }
        resolve();
      }
    }, 12000); // Increased timeout
  });
}

async function generateThumbnail(imagePath, thumbnailPath) {
  try {
    await sharp(imagePath)
      .resize(300, 300, { fit: 'cover' })
      .jpeg({ quality: 80 })
      .toFile(thumbnailPath);
    return thumbnailPath;
  } catch (error) {
    console.error('Error generating thumbnail:', error);
    return null;
  }
}

async function extractMetadata(imagePath) {
  try {
    const exifData = await exifr.parse(imagePath);
    const stats = await fs.stat(imagePath);
    
    let timestamp = null;
    if (exifData && exifData.DateTimeOriginal) {
      timestamp = new Date(exifData.DateTimeOriginal);
    } else if (exifData && exifData.DateTime) {
      timestamp = new Date(exifData.DateTime);
    } else {
      timestamp = stats.birthtime || stats.mtime;
    }

    return {
      filename: path.basename(imagePath),
      path: imagePath,
      timestamp: timestamp,
      fileSize: stats.size,
      exif: exifData || {},
      createdAt: new Date()
    };
  } catch (error) {
    console.error('Error extracting metadata:', error);
    return null;
  }
}

async function detectFaces(imagePath, photoId) {
  try {
    console.log(`DEBUG: Starting face detection for photo ${photoId} at ${imagePath}`);
    console.log(`DEBUG: Using face service URL: ${FACE_SERVICE_URL}`);
    
    const form = new FormData();
    const imageBuffer = await fs.readFile(imagePath);
    form.append('file', imageBuffer, path.basename(imagePath));
    form.append('photo_id', photoId);
    form.append('photo_path', imagePath);

    console.log(`DEBUG: Sending request to ${FACE_SERVICE_URL}/detect-faces`);
    const response = await axios.post(`${FACE_SERVICE_URL}/detect-faces`, form, {
      headers: form.getHeaders(),
      timeout: 120000  // Increase timeout to 2 minutes
    });

    console.log(`DEBUG: Face detection response:`, response.data);
    return response.data;
  } catch (error) {
    console.error('ERROR: Face detection failed:', error.message);
    if (error.response) {
      console.error('ERROR: Response status:', error.response.status);
      console.error('ERROR: Response data:', error.response.data);
    }
    return { success: false, faces: [] };
  }
}

ipcMain.handle('upload-photos', async (event, filePaths) => {
  const results = [];
  let processedCount = 0;
  
  for (const filePath of filePaths) {
    try {
      const originalFilename = path.basename(filePath);
      const fileExtension = path.extname(originalFilename);
      
      // Generate unique ID for this photo (UTF-8 safe)
      const photoId = Date.now().toString() + Math.random().toString(36).substr(2, 9);
      const safeFilename = `${photoId}${fileExtension}`;
      const newPath = path.join(PHOTOS_DIR, safeFilename);
      const thumbnailPath = path.join(THUMBNAILS_DIR, `thumb_${safeFilename}`);
      
      await fs.copyFile(filePath, newPath);
      
      const metadata = await extractMetadata(newPath);
      if (metadata) {
        // Store original filename for display purposes
        metadata.originalFilename = originalFilename;
        metadata.filename = safeFilename;  // Use safe filename for storage
        metadata.id = photoId;  // Explicit ID assignment
        metadata.thumbnailPath = await generateThumbnail(newPath, thumbnailPath);
        
        // Save to database
        const savedPhoto = await addPhotoToDB(metadata);
        if (savedPhoto) {
          results.push(savedPhoto);
          
          // Send progress update
          processedCount++;
          const progress = Math.round((processedCount / filePaths.length) * 50); // 50% for upload
          event.sender.send('upload-progress', { progress, phase: 'uploading', current: processedCount, total: filePaths.length });
        }
      }
    } catch (error) {
      console.error('Error processing file:', filePath, error);
    }
  }
  
  // Now process faces
  event.sender.send('upload-progress', { progress: 50, phase: 'detecting-faces', current: 0, total: results.length });
  
  let facesProcessed = 0;
  for (const photo of results) {
    try {
      const faceResult = await detectFaces(photo.path, photo.id);
      if (faceResult.success && faceResult.faces.length > 0) {
        await addFacesToDB(faceResult.faces, photo.id);
      }
      
      facesProcessed++;
      const progress = 50 + Math.round((facesProcessed / results.length) * 50); // 50-100% for face detection
      event.sender.send('upload-progress', { progress, phase: 'detecting-faces', current: facesProcessed, total: results.length });
    } catch (error) {
      console.error('Face detection error:', error);
      facesProcessed++;
      const progress = 50 + Math.round((facesProcessed / results.length) * 50);
      event.sender.send('upload-progress', { progress, phase: 'detecting-faces', current: facesProcessed, total: results.length });
    }
  }
  
  // Complete
  event.sender.send('upload-progress', { progress: 100, phase: 'complete', current: results.length, total: results.length });
  
  return results;
});

ipcMain.handle('get-photos', async () => {
  try {
    const photos = await getAllPhotos();
    return photos;
  } catch (error) {
    console.error('Error fetching photos:', error);
    return [];
  }
});

ipcMain.handle('show-file-dialog', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile', 'multiSelections'],
    filters: [
      { name: 'Images', extensions: ['jpg', 'jpeg', 'png'] }
    ]
  });
  
  return result.filePaths;
});

ipcMain.handle('get-face-clusters', async () => {
  try {
    console.log(`DEBUG: Requesting face clusters from ${FACE_SERVICE_URL}/face-clusters`);
    const response = await axios.get(`${FACE_SERVICE_URL}/face-clusters`);
    console.log('DEBUG: Face clusters response:', response.data);
    return response.data;
  } catch (error) {
    console.error('ERROR: Getting face clusters failed:', error.message);
    if (error.response) {
      console.error('ERROR: Response status:', error.response.status);
      console.error('ERROR: Response data:', error.response.data);
    }
    return { success: false, clusters: [] };
  }
});

ipcMain.handle('find-similar-faces', async (event, imagePath) => {
  try {
    const form = new FormData();
    const imageBuffer = await fs.readFile(imagePath);
    form.append('file', imageBuffer, path.basename(imagePath));

    const response = await axios.post(`${FACE_SERVICE_URL}/find-similar-faces`, form, {
      headers: form.getHeaders(),
      timeout: 120000  // Increase timeout to 2 minutes
    });

    return response.data;
  } catch (error) {
    console.error('Error finding similar faces:', error);
    return { success: false, similar_faces: [] };
  }
});

ipcMain.handle('delete-photo', async (event, photoId) => {
  try {
    console.log(`Deleting photo: ${photoId}`);
    
    // Delete from database and get photo info
    const deleteResult = await deletePhotoFromDB(photoId);
    
    if (!deleteResult.success) {
      return { success: false, message: deleteResult.message };
    }
    
    // Clean up files
    const filesDeleted = await cleanupPhotoFiles(deleteResult.photo);
    
    // Also remove from face detection service metadata
    try {
      // This is a best-effort cleanup - face service might not be running
      await axios.delete(`${FACE_SERVICE_URL}/photos/${photoId}`);
    } catch (error) {
      console.log('Face service cleanup skipped (service may be offline)');
    }
    
    console.log(`Successfully deleted photo ${photoId} and ${filesDeleted} associated files`);
    return { success: true, filesDeleted };
    
  } catch (error) {
    console.error('Error deleting photo:', error);
    return { success: false, message: error.message };
  }
});

ipcMain.handle('delete-photos', async (event, photoIds) => {
  try {
    console.log(`Batch deleting ${photoIds.length} photos`);
    
    const results = {
      success: true,
      deleted: 0,
      failed: 0,
      errors: []
    };
    
    for (const photoId of photoIds) {
      try {
        // Send progress update
        event.sender.send('delete-progress', { 
          current: results.deleted + results.failed + 1, 
          total: photoIds.length,
          phase: 'deleting'
        });
        
        const deleteResult = await deletePhotoFromDB(photoId);
        
        if (deleteResult.success) {
          await cleanupPhotoFiles(deleteResult.photo);
          results.deleted++;
        } else {
          results.failed++;
          results.errors.push(`Failed to delete ${photoId}: ${deleteResult.message}`);
        }
        
        // Clean up from face service (best effort)
        try {
          await axios.delete(`${FACE_SERVICE_URL}/photos/${photoId}`);
        } catch (error) {
          // Ignore face service errors
        }
        
      } catch (error) {
        results.failed++;
        results.errors.push(`Error deleting ${photoId}: ${error.message}`);
      }
    }
    
    // Send completion
    event.sender.send('delete-progress', { 
      current: photoIds.length, 
      total: photoIds.length,
      phase: 'complete'
    });
    
    console.log(`Batch delete completed: ${results.deleted} deleted, ${results.failed} failed`);
    return results;
    
  } catch (error) {
    console.error('Error in batch delete:', error);
    return { success: false, message: error.message };
  }
});

// Disable hardware acceleration to prevent GPU process errors (optional)
app.disableHardwareAcceleration();

app.whenReady().then(async () => {
  await initializeDirectories();
  await initializeDatabase();
  
  // Start face detection service
  try {
    await startFaceDetectionService();
  } catch (error) {
    console.error('Face detection service failed to start:', error);
    // Continue without face detection
  }
  
  await createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    cleanup();
    app.quit();
  }
});

app.on('before-quit', (event) => {
  event.preventDefault();
  cleanup();
  setTimeout(() => {
    app.exit();
  }, 1000);
});

function cleanup() {
  // Close database properly
  if (db) {
    try {
      db.close((err) => {
        if (err) {
          console.error('Error closing database:', err);
        } else {
          console.log('Database connection closed');
        }
      });
      db = null;
    } catch (error) {
      console.error('Database cleanup error:', error);
    }
  }
  
  // Terminate Python processes
  if (pythonProcess) {
    try {
      console.log('Terminating face detection service...');
      pythonProcess.kill('SIGTERM');
      pythonProcess = null;
    } catch (error) {
      console.error('Python process cleanup error:', error);
    }
  }
  
  // Additional cleanup: kill any remaining Python processes on Windows
  if (process.platform === 'win32') {
    try {
      const { exec } = require('child_process');
      exec('taskkill //F //IM python.exe 2>/dev/null', (error, stdout, stderr) => {
        if (!error) {
          console.log('Additional Python processes terminated');
        }
      });
    } catch (error) {
      console.error('Additional cleanup error:', error);
    }
  }
}