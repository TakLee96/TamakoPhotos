class PhotoApp {
    constructor() {
        this.currentView = 'timeline';
        this.photos = [];
        this.faceClusters = [];
        this.processingCount = 0;
        this.faceServiceAvailable = true;
        this.selectionMode = false;
        this.selectedPhotos = new Set();
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadPhotos();
        // Check face service on startup
        setTimeout(() => this.checkFaceServiceStatus(), 2000);
        
        // Listen for upload progress
        window.electronAPI.onUploadProgress((event, data) => {
            this.updateProgress(data);
        });
        
        // Listen for delete progress
        window.electronAPI.onDeleteProgress((event, data) => {
            this.updateDeleteProgress(data);
        });
    }

    setupEventListeners() {
        const timelineBtn = document.getElementById('timeline-btn');
        const facesBtn = document.getElementById('faces-btn');
        const uploadBtn = document.getElementById('upload-btn');
        const selectBtn = document.getElementById('select-btn');
        const selectAllBtn = document.getElementById('select-all-btn');
        const deleteSelectedBtn = document.getElementById('delete-selected-btn');
        const cancelSelectionBtn = document.getElementById('cancel-selection-btn');

        timelineBtn.addEventListener('click', () => this.switchView('timeline'));
        facesBtn.addEventListener('click', () => this.switchView('faces'));
        uploadBtn.addEventListener('click', () => this.handleUpload());
        selectBtn.addEventListener('click', () => this.toggleSelectionMode());
        selectAllBtn.addEventListener('click', () => this.selectAllPhotos());
        deleteSelectedBtn.addEventListener('click', () => this.deleteSelectedPhotos());
        cancelSelectionBtn.addEventListener('click', () => this.cancelSelection());

        // Add keyboard shortcuts
        document.addEventListener('keydown', (e) => this.handleKeyboardShortcuts(e));

        this.setupDragAndDrop();
    }

    setupDragAndDrop() {
        const dropZone = document.getElementById('drop-zone');
        const body = document.body;

        body.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.remove('hidden');
        });

        body.addEventListener('dragleave', (e) => {
            if (e.clientX === 0 && e.clientY === 0) {
                dropZone.classList.add('hidden');
            }
        });

        body.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.add('hidden');
            
            const files = Array.from(e.dataTransfer.files);
            const imagePaths = files
                .filter(file => ['image/jpeg', 'image/jpg', 'image/png'].includes(file.type))
                .map(file => file.path);
            
            if (imagePaths.length > 0) {
                this.uploadPhotos(imagePaths);
            }
        });

        dropZone.addEventListener('dragover', (e) => e.preventDefault());
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.add('hidden');
        });
    }

    switchView(view) {
        document.querySelectorAll('.nav-btn').forEach(btn => btn.classList.remove('active'));
        document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));

        document.getElementById(`${view}-btn`).classList.add('active');
        document.getElementById(`${view}-view`).classList.add('active');

        this.currentView = view;

        if (view === 'faces') {
            this.loadFaceClusters();
        }
    }

    async handleUpload() {
        try {
            const filePaths = await window.electronAPI.showFileDialog();
            if (filePaths && filePaths.length > 0) {
                await this.uploadPhotos(filePaths);
            }
        } catch (error) {
            console.error('Error handling upload:', error);
        }
    }

    async uploadPhotos(filePaths) {
        try {
            const photoGrid = document.getElementById('photo-grid');
            photoGrid.innerHTML = '<div class="loading">Uploading photos...</div>';

            // Show processing status
            this.processingCount = filePaths.length;
            this.showProcessingStatus();

            const uploadedPhotos = await window.electronAPI.uploadPhotos(filePaths);
            this.photos = [...uploadedPhotos, ...this.photos];
            this.renderPhotos();

            // Check face service status
            this.checkFaceServiceStatus();
        } catch (error) {
            console.error('Error uploading photos:', error);
            const photoGrid = document.getElementById('photo-grid');
            photoGrid.innerHTML = '<div class="loading">Error uploading photos</div>';
            this.hideProcessingStatus();
        }
    }

    showProcessingStatus() {
        const statusBar = document.getElementById('status-bar');
        const statusText = document.querySelector('.status-text');
        statusText.textContent = `Processing ${this.processingCount} photo${this.processingCount > 1 ? 's' : ''} for faces...`;
        statusBar.classList.remove('hidden');
    }

    updateProgress(data) {
        const statusBar = document.getElementById('status-bar');
        const statusText = document.querySelector('.status-text');
        const progressFill = document.querySelector('.progress-fill');
        
        if (data.phase === 'uploading') {
            statusText.textContent = `Uploading ${data.current}/${data.total} photos...`;
        } else if (data.phase === 'detecting-faces') {
            statusText.textContent = `Detecting faces in ${data.current}/${data.total} photos...`;
        } else if (data.phase === 'complete') {
            statusText.textContent = `Completed processing ${data.total} photos!`;
            setTimeout(() => {
                this.hideProcessingStatus();
                this.loadPhotos(); // Refresh photos to show updated face data
            }, 2000);
        }
        
        // Update progress bar
        progressFill.style.width = `${data.progress}%`;
        progressFill.style.animation = 'none'; // Remove fake animation
        
        statusBar.classList.remove('hidden');
    }

    hideProcessingStatus() {
        const statusBar = document.getElementById('status-bar');
        statusBar.classList.add('hidden');
        this.processingCount = 0;
        
        // Reset progress bar
        const progressFill = document.querySelector('.progress-fill');
        progressFill.style.width = '0%';
        progressFill.style.animation = 'progress-pulse 2s ease-in-out infinite';
    }

    showError(message) {
        const errorBar = document.getElementById('error-bar');
        const errorText = document.querySelector('.error-text');
        errorText.textContent = message;
        errorBar.classList.remove('hidden');
    }

    dismissError() {
        const errorBar = document.getElementById('error-bar');
        errorBar.classList.add('hidden');
    }

    async checkFaceServiceStatus() {
        try {
            const response = await fetch('http://127.0.0.1:8000/stats');
            if (response.ok) {
                this.faceServiceAvailable = true;
                // Hide any existing error
                this.dismissError();
            } else {
                throw new Error('Service unavailable');
            }
        } catch (error) {
            this.faceServiceAvailable = false;
            this.showError('Face detection service unavailable - photo management still works!');
        }
    }

    async loadPhotos() {
        try {
            this.photos = await window.electronAPI.getPhotos();
            this.renderPhotos();
        } catch (error) {
            console.error('Error loading photos:', error);
            const photoGrid = document.getElementById('photo-grid');
            photoGrid.innerHTML = '<div class="loading">Error loading photos</div>';
        }
    }

    renderPhotos() {
        const photoGrid = document.getElementById('photo-grid');
        
        if (this.photos.length === 0) {
            photoGrid.innerHTML = `
                <div class="empty-state">
                    <h3>No photos yet</h3>
                    <p>Upload some photos to get started</p>
                    <button onclick="app.handleUpload()">Upload Photos</button>
                </div>
            `;
            return;
        }

        photoGrid.innerHTML = this.photos.map(photo => {
            const date = new Date(photo.timestamp).toLocaleDateString('en-US', {
                year: 'numeric',
                month: 'short',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
            });

            const thumbnailSrc = photo.thumbnailPath 
                ? `file://${photo.thumbnailPath.replace(/\\/g, '/')}`
                : `file://${photo.path.replace(/\\/g, '/')}`;

            return `
                <div class="photo-item" data-photo-id="${photo.id}" onclick="app.openPhoto('${photo.id}')">
                    <button class="photo-delete-btn" onclick="event.stopPropagation(); app.deletePhoto('${photo.id}')" title="Delete photo">×</button>
                    <div class="photo-checkbox" onclick="app.togglePhotoSelection('${photo.id}', event)"></div>
                    <img src="${thumbnailSrc}" alt="${photo.originalFilename || photo.filename}" loading="lazy">
                    <div class="photo-info">
                        <div class="photo-date">${date}</div>
                        <div class="photo-name">${photo.originalFilename || photo.filename}</div>
                    </div>
                </div>
            `;
        }).join('');
    }

    async loadFaceClusters() {
        try {
            const facesGrid = document.getElementById('faces-grid');
            facesGrid.innerHTML = '<div class="loading">Loading face clusters...</div>';

            const result = await window.electronAPI.getFaceClusters();
            if (result.success) {
                // Sort clusters by face count in descending order (largest clusters first)
                this.faceClusters = result.clusters.sort((a, b) => b.face_count - a.face_count);
                this.renderFaceClusters();
            } else {
                facesGrid.innerHTML = '<div class="loading">Error loading face clusters</div>';
            }
        } catch (error) {
            console.error('Error loading face clusters:', error);
            document.getElementById('faces-grid').innerHTML = '<div class="loading">Error loading face clusters</div>';
        }
    }

    renderFaceClusters() {
        const facesGrid = document.getElementById('faces-grid');
        
        if (this.faceClusters.length === 0) {
            facesGrid.innerHTML = `
                <div class="empty-state">
                    <h3>No face clusters found</h3>
                    <p>Upload some photos with faces to see them grouped here</p>
                </div>
            `;
            return;
        }

        facesGrid.innerHTML = this.faceClusters.map((cluster, index) => {
            const firstFace = cluster.faces[0];
            const photoText = cluster.face_count === 1 ? 'photo' : 'photos';
            return `
                <div class="face-cluster" onclick="app.viewCluster(${cluster.cluster_id})">
                    <div class="cluster-preview">
                        <img src="data:image/jpeg;base64,${firstFace.face_image}" alt="Face" class="face-thumbnail">
                        <div class="face-count">${cluster.face_count} ${photoText}</div>
                    </div>
                    <div class="cluster-info">
                        <div class="cluster-title">Person ${index + 1}</div>
                    </div>
                </div>
            `;
        }).join('');
    }

    async viewCluster(clusterId) {
        const cluster = this.faceClusters.find(c => c.cluster_id === clusterId);
        if (!cluster) return;

        // Get all photo IDs for faces in this cluster
        const photoIds = [...new Set(cluster.faces.map(face => face.photo_id))];
        
        // Get photo details for these IDs
        const clusterPhotos = this.photos.filter(photo => photoIds.includes(photo.id));
        
        // Create and show cluster modal
        this.showClusterModal(cluster, clusterPhotos);
    }

    showClusterModal(cluster, photos) {
        // Remove existing modal if any
        const existingModal = document.getElementById('cluster-modal');
        if (existingModal) {
            existingModal.remove();
        }

        // Create modal HTML
        const modal = document.createElement('div');
        modal.id = 'cluster-modal';
        modal.className = 'modal';
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h2>Face Cluster - ${cluster.face_count} ${cluster.face_count === 1 ? 'photo' : 'photos'}</h2>
                    <button class="modal-close" onclick="app.closeClusterModal()">&times;</button>
                </div>
                <div class="modal-body">
                    <div class="cluster-faces">
                        <h3>Detected Faces</h3>
                        <div class="faces-preview">
                            ${cluster.faces.slice(0, 6).map(face => `
                                <div class="face-preview">
                                    <img src="data:image/jpeg;base64,${face.face_image}" alt="Face">
                                    <div class="face-confidence">${(face.confidence * 100).toFixed(0)}%</div>
                                </div>
                            `).join('')}
                            ${cluster.faces.length > 6 ? `<div class="face-preview more">+${cluster.faces.length - 6} more</div>` : ''}
                        </div>
                    </div>
                    <div class="cluster-photos">
                        <h3>Photos (${photos.length})</h3>
                        <div class="cluster-photo-grid">
                            ${photos.map(photo => {
                                const thumbnailSrc = photo.thumbnailPath ? 
                                    `file://${photo.thumbnailPath.replace(/\\/g, '/')}` : 
                                    `file://${photo.path.replace(/\\/g, '/')}`;
                                const date = photo.timestamp ? new Date(photo.timestamp).toLocaleDateString() : 'Unknown date';
                                
                                return `
                                    <div class="cluster-photo-item" onclick="app.openPhoto('${photo.id}')">
                                        <img src="${thumbnailSrc}" alt="${photo.originalFilename || photo.filename}" loading="lazy">
                                        <div class="cluster-photo-info">
                                            <div class="cluster-photo-date">${date}</div>
                                            <div class="cluster-photo-name">${photo.originalFilename || photo.filename}</div>
                                        </div>
                                    </div>
                                `;
                            }).join('')}
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Add modal to page
        document.body.appendChild(modal);

        // Show modal with animation
        setTimeout(() => modal.classList.add('show'), 10);

        // Close on background click
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                this.closeClusterModal();
            }
        });
    }

    closeClusterModal() {
        const modal = document.getElementById('cluster-modal');
        if (modal) {
            modal.classList.remove('show');
            setTimeout(() => modal.remove(), 300);
        }
    }

    toggleSelectionMode() {
        this.selectionMode = !this.selectionMode;
        this.selectedPhotos.clear();
        
        const selectBtn = document.getElementById('select-btn');
        const selectAllBtn = document.getElementById('select-all-btn');
        const deleteSelectedBtn = document.getElementById('delete-selected-btn');
        const cancelSelectionBtn = document.getElementById('cancel-selection-btn');
        const photoGrid = document.getElementById('photo-grid');
        
        if (this.selectionMode) {
            selectBtn.textContent = 'Cancel';
            selectAllBtn.classList.remove('hidden');
            deleteSelectedBtn.classList.remove('hidden');
            cancelSelectionBtn.classList.remove('hidden');
            photoGrid.classList.add('selecting');
        } else {
            selectBtn.textContent = 'Select';
            selectAllBtn.classList.add('hidden');
            deleteSelectedBtn.classList.add('hidden');
            cancelSelectionBtn.classList.add('hidden');
            photoGrid.classList.remove('selecting');
        }
        
        this.updateSelectionUI();
    }
    
    cancelSelection() {
        this.selectionMode = false;
        this.selectedPhotos.clear();
        this.toggleSelectionMode();
    }
    
    togglePhotoSelection(photoId, event) {
        event.stopPropagation();
        
        if (this.selectedPhotos.has(photoId)) {
            this.selectedPhotos.delete(photoId);
        } else {
            this.selectedPhotos.add(photoId);
        }
        
        this.updateSelectionUI();
    }
    
    updateSelectionUI() {
        const deleteSelectedBtn = document.getElementById('delete-selected-btn');
        const count = this.selectedPhotos.size;
        
        if (count > 0) {
            deleteSelectedBtn.textContent = `Delete Selected (${count})`;
            deleteSelectedBtn.disabled = false;
        } else {
            deleteSelectedBtn.textContent = 'Delete Selected';
            deleteSelectedBtn.disabled = true;
        }
        
        // Update checkboxes
        document.querySelectorAll('.photo-item').forEach(item => {
            const photoId = item.dataset.photoId;
            const checkbox = item.querySelector('.photo-checkbox');
            
            if (this.selectedPhotos.has(photoId)) {
                item.classList.add('selected');
                checkbox.classList.add('checked');
            } else {
                item.classList.remove('selected');
                checkbox.classList.remove('checked');
            }
        });
    }
    
    async deletePhoto(photoId) {
        if (!confirm('Are you sure you want to delete this photo? This action cannot be undone.')) {
            return;
        }
        
        try {
            const result = await window.electronAPI.deletePhoto(photoId);
            
            if (result.success) {
                // Remove from local array
                this.photos = this.photos.filter(p => p.id !== photoId);
                this.renderPhotos();
                
                this.showStatus(`Photo deleted successfully`, 'success');
            } else {
                this.showError(`Failed to delete photo: ${result.message}`);
            }
        } catch (error) {
            console.error('Error deleting photo:', error);
            this.showError('Failed to delete photo');
        }
    }
    
    async deleteSelectedPhotos() {
        const count = this.selectedPhotos.size;
        if (count === 0) return;
        
        const message = `Are you sure you want to delete ${count} photo${count > 1 ? 's' : ''}? This action cannot be undone.`;
        if (!confirm(message)) return;
        
        try {
            const photoIds = Array.from(this.selectedPhotos);
            this.showProcessingStatus();
            
            const result = await window.electronAPI.deletePhotos(photoIds);
            
            if (result.success) {
                // Remove deleted photos from local array
                this.photos = this.photos.filter(p => !photoIds.includes(p.id));
                this.selectedPhotos.clear();
                this.cancelSelection();
                this.renderPhotos();
                
                this.showStatus(`Successfully deleted ${result.deleted} photo${result.deleted > 1 ? 's' : ''}`, 'success');
                
                if (result.failed > 0) {
                    console.error('Some deletions failed:', result.errors);
                    this.showError(`${result.failed} photo${result.failed > 1 ? 's' : ''} could not be deleted`);
                }
            } else {
                this.showError(`Failed to delete photos: ${result.message}`);
            }
        } catch (error) {
            console.error('Error deleting photos:', error);
            this.showError('Failed to delete photos');
        } finally {
            this.hideProcessingStatus();
        }
    }
    
    updateDeleteProgress(data) {
        const statusBar = document.getElementById('status-bar');
        const statusText = document.querySelector('.status-text');
        const progressFill = document.querySelector('.progress-fill');
        
        if (data.phase === 'deleting') {
            statusText.textContent = `Deleting ${data.current}/${data.total} photos...`;
            const progress = Math.round((data.current / data.total) * 100);
            progressFill.style.width = `${progress}%`;
        } else if (data.phase === 'complete') {
            statusText.textContent = `Deletion completed!`;
            progressFill.style.width = '100%';
            setTimeout(() => {
                this.hideProcessingStatus();
            }, 1500);
        }
        
        statusBar.classList.remove('hidden');
    }
    
    showStatus(message, type = 'info') {
        const statusBar = document.getElementById('status-bar');
        const statusText = document.querySelector('.status-text');
        
        statusText.textContent = message;
        statusBar.className = `status-bar ${type}`;
        statusBar.classList.remove('hidden');
        
        setTimeout(() => {
            statusBar.classList.add('hidden');
        }, 3000);
    }

    selectAllPhotos() {
        if (!this.selectionMode || this.photos.length === 0) return;
        
        // Select all visible photos
        this.photos.forEach(photo => {
            this.selectedPhotos.add(photo.id);
        });
        
        this.updateSelectionUI();
    }
    
    handleKeyboardShortcuts(e) {
        // CTRL + A to select all (only in selection mode and timeline view)
        if (e.ctrlKey && e.key === 'a' && this.selectionMode && this.currentView === 'timeline') {
            e.preventDefault();
            this.selectAllPhotos();
        }
        
        // ESC to cancel selection mode
        if (e.key === 'Escape' && this.selectionMode) {
            this.cancelSelection();
        }
    }

    openPhoto(photoId) {
        if (this.selectionMode) {
            this.togglePhotoSelection(photoId, { stopPropagation: () => {} });
        } else {
            console.log('Opening photo:', photoId);
        }
    }
}

const app = new PhotoApp();