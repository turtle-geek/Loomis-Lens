const API_URL = "https://2w7jkxljirnws5pgbp2peyiske0ymbls.lambda-url.us-east-2.on.aws/generate-overlay";

// Select all necessary DOM elements for image processing and UI control.
const webcamBtn = document.getElementById('webcamBtn');
const uploadBtn = document.getElementById('uploadBtn');
const fileInput = document.getElementById('fileInput');
const webcamVideo = document.getElementById('webcamVideo');
const baseImage = document.getElementById('baseImage');
const resultImage = document.getElementById('resultImage');
const captureCanvas = document.getElementById('captureCanvas');
const loading = document.getElementById('loading');
const opacitySlider = document.getElementById('opacitySlider');
const opacityLabel = document.getElementById('opacityLabel');

// Track the state of the webcam to toggle between 'Live' and 'Capture' modes.
let isWebcamLive = false;

/**
 * Updates the custom range slider styling and the percentage text display.
 * @param {number} val: The current value of the opacity slider (0-100).
 */
function updateSliderUI(val) {
    opacitySlider.style.background = `linear-gradient(to right, #2563eb 0%, #2563eb ${val}%, #f3f4f6 ${val}%, #f3f4f6 100%)`;
    opacityLabel.innerText = `${val}%`;
}

/**
 * Stops all active video tracks and resets the webcam button to its original state.
 */
function stopCamera() {
    if (webcamVideo.srcObject) {
        const tracks = webcamVideo.srcObject.getTracks();
        tracks.forEach(track => track.stop());
        webcamVideo.srcObject = null;
    }
    webcamVideo.classList.add('hidden');
    isWebcamLive = false;
    webcamBtn.innerHTML = `
        <svg xmlns="http://www.w3.org/2000/svg" class="w-6 h-6 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
        </svg>
        Live Webcam`;
    webcamBtn.classList.replace('bg-red-600', 'bg-green-600');
}

// Handle the webcam button click for both starting the camera and capturing snapshots.
webcamBtn.addEventListener('click', async () => {
    if (!isWebcamLive) {
        // Mode: Start the camera stream and prepare the UI for a snapshot.
        try {
            baseImage.src = "";
            resultImage.src = "";
            baseImage.classList.add('hidden');
            resultImage.classList.add('hidden');
            
            const stream = await navigator.mediaDevices.getUserMedia({ 
                video: { width: { ideal: 1280 }, height: { ideal: 720 } }, 
                audio: false 
            });
            webcamVideo.srcObject = stream;
            webcamVideo.classList.remove('hidden');
            isWebcamLive = true;
            webcamBtn.innerText = "Capture Snapshot";
            webcamBtn.classList.replace('bg-green-600', 'bg-red-600');
        } catch (err) {
            alert("Camera access was denied or is unavailable: " + err.message);
        }
    } else {
        // Mode: Capture a frame from the video stream and send it to the backend.
        const context = captureCanvas.getContext('2d');
        captureCanvas.width = webcamVideo.videoWidth;
        captureCanvas.height = webcamVideo.videoHeight;
        
        // Translate and scale the canvas to match the mirrored webcam preview.
        context.translate(captureCanvas.width, 0);
        context.scale(-1, 1);
        context.drawImage(webcamVideo, 0, 0, captureCanvas.width, captureCanvas.height);

        captureCanvas.toBlob(async (blob) => {
            const file = new File([blob], "webcam.png", { type: "image/png" });
            stopCamera(); 
            await uploadAndProcess(file);
        }, 'image/png');
    }
});

// Reset the view and trigger the hidden file input when the upload button is clicked.
uploadBtn.addEventListener('click', () => {
    stopCamera(); 
    baseImage.classList.add('hidden');
    resultImage.classList.add('hidden');
    fileInput.value = ""; 
    fileInput.click();
});

// Process the file immediately after a user selects an image from their device.
fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (file) await uploadAndProcess(file);
});

/**
 * Uploads the image to AWS Lambda and manages the two-stage rendering process.
 * @param {File} file: The image file to be processed by the AI model.
 */
async function uploadAndProcess(file) {
    // Stage 1: Display the original face on the base layer immediately for zero-latency feedback.
    const reader = new FileReader();
    reader.onload = (e) => {
        baseImage.src = e.target.result;
        baseImage.classList.remove('hidden');
        resultImage.classList.add('hidden'); // Ensure the previous overlay is hidden.
    };
    reader.readAsDataURL(file);

    // Activate the loading spinner while the AI model processes the image.
    loading.classList.remove('hidden');
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('The API returned an error during processing.');

        const blob = await response.blob();
        const imageUrl = URL.createObjectURL(blob);
        
        // Stage 2: Render the transparent Loomis lines on the top layer.
        resultImage.src = imageUrl;
        resultImage.classList.remove('hidden');
        resultImage.style.opacity = opacitySlider.value / 100;
    } catch (err) {
        alert("An error occurred: " + err.message);
    } finally {
        loading.classList.add('hidden');
    }
}

// Adjust the transparency of the overlay lines without affecting the base portrait.
opacitySlider.addEventListener('input', (e) => {
    const val = e.target.value;
    resultImage.style.opacity = val / 100;
    updateSliderUI(val);
});

// Clear all images, reset the UI, and ensure the camera is stopped.
document.getElementById('resetView').addEventListener('click', () => {
    stopCamera();
    baseImage.src = "";
    resultImage.src = "";
    baseImage.classList.add('hidden');
    resultImage.classList.add('hidden');
    fileInput.value = "";
    opacitySlider.value = 100;
    updateSliderUI(100);
});