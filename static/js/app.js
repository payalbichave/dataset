document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const removeBtn = document.getElementById('remove-btn');
    const analyzeBtn = document.getElementById('analyze-btn');
    
    const uploadSection = document.getElementById('upload-section');
    const loadingState = document.getElementById('loading-state');
    const resultSection = document.getElementById('result-section');
    const resetBtn = document.getElementById('reset-btn');
    
    // Result Elements
    const diseaseName = document.getElementById('disease-name');
    const confidenceValue = document.getElementById('confidence-value');
    const progressFill = document.getElementById('progress-fill');
    const statusIcon = document.getElementById('status-icon');
    const recommendationText = document.getElementById('recommendation-text');
    
    let selectedFile = null;

    // --- Drag and Drop Logic --- //
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    });

    // --- Click Upload Logic --- //
    browseBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });

    function handleFiles(files) {
        if (files.length === 0) return;
        
        const file = files[0];
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file.');
            return;
        }

        selectedFile = file;
        
        // Show preview
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = () => {
            imagePreview.src = reader.result;
            document.querySelector('.drop-zone-content').classList.add('hidden');
            previewContainer.classList.remove('hidden');
            analyzeBtn.classList.remove('hidden');
        };
    }

    // --- Remove Image --- //
    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation(); // prevent triggering dropzone click if any
        selectedFile = null;
        fileInput.value = '';
        imagePreview.src = '';
        previewContainer.classList.add('hidden');
        document.querySelector('.drop-zone-content').classList.remove('hidden');
        analyzeBtn.classList.add('hidden');
    });

    // --- Analyze Button Logic --- //
    analyzeBtn.addEventListener('click', () => {
        if (!selectedFile) return;

        // Transition to loading
        uploadSection.classList.add('hidden');
        loadingState.classList.remove('hidden');

        // Create form data
        const formData = new FormData();
        formData.append('file', selectedFile);

        // Send to backend
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Artificial delay to show the nice animation and give a sense of processing
            setTimeout(() => {
                showResults(data);
            }, 1200);
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred during analysis. Make sure the Flask server is running.');
            resetUI();
        });
    });

    // --- Display Results --- //
    function showResults(data) {
        loadingState.classList.add('hidden');
        resultSection.classList.remove('hidden');

        if (data.error) {
            diseaseName.textContent = "Error during analysis";
            confidenceValue.textContent = "N/A";
            progressFill.style.width = "0%";
            recommendationText.textContent = data.error;
            return;
        }

        const conf = data.confidence.toFixed(1);
        
        diseaseName.textContent = data.class;
        confidenceValue.textContent = `${conf}%`;
        
        // Set progress bar width and color
        progressFill.style.width = '0%';
        setTimeout(() => {
            progressFill.style.width = `${conf}%`;
        }, 100);

        // Determine health status style
        statusIcon.className = 'status-icon';
        progressFill.className = 'progress-fill';
        
        if (data.is_healthy) {
            statusIcon.classList.add('healthy');
            statusIcon.innerHTML = '<i class="fa-solid fa-leaf"></i>';
            progressFill.classList.add('high');
            recommendationText.innerHTML = "Your plant appears to be <strong>healthy</strong>! Continue your current watering and lighting schedule.";
        } else {
            // Sick plant
            statusIcon.classList.add('sick');
            statusIcon.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i>';
            
            if (conf >= 80) progressFill.classList.add('low');
            else progressFill.classList.add('medium');
            
            recommendationText.innerHTML = `This plant shows signs of <strong>${data.class.toLowerCase()}</strong>. Consider isolating this plant and treating the affected areas appropriately based on the specific disease.`;
        }
    }

    // --- Reset UI --- //
    resetBtn.addEventListener('click', resetUI);

    function resetUI() {
        resultSection.classList.add('hidden');
        loadingState.classList.add('hidden');
        uploadSection.classList.remove('hidden');
        
        removeBtn.click(); // Clears selection
    }
});
