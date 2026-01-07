const API_URL = "https://2w7jkxljirnws5pgbp2peyiske0ymbls.lambda-url.us-east-2.on.aws/generate-overlay";

const webcamBtn = document.getElementById('webcamBtn');
const uploadBtn = document.getElementById('uploadBtn');
const fileInput = document.getElementById('fileInput');
const webcamVideo = document.getElementById('webcamVideo');
const resultImage = document.getElementById('resultImage');
const captureCanvas = document.getElementById('captureCanvas');
const loading = document.getElementById('loading');
const opacitySlider = document.getElementById('opacitySlider');

let isWebcamLive = false;

// This function properly shuts down the camera stream and releases the hardware
function stopCamera() {
    if (webcamVideo.srcObject) {
        const tracks = webcamVideo.srcObject.getTracks();
        tracks.forEach(track => track.stop());
        webcamVideo.srcObject = null;
    }
    webcamVideo.classList.add('hidden');
    isWebcamLive = false;
    
    // Reset the button back to its original green appearance
    webcamBtn.innerHTML = `
        <svg xmlns="http://www.w3.org/2000/svg" class="w-6 h-6 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
        </svg>
        Live Webcam`;
    webcamBtn.classList.replace('bg-red-600', 'bg-green-600');
}

webcamBtn.addEventListener('click', async () => {
    if (!isWebcamLive) {
        try {
            // Clear any old images before starting the webcam
            resultImage.src = "";
            resultImage.classList.add('hidden');

            // Request high definition camera access without audio
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
            alert("Camera access denied: " + err.message);
        }
    } else {
        // Prepare the canvas for taking a snapshot
        const context = captureCanvas.getContext('2d');
        captureCanvas.width = webcamVideo.videoWidth;
        captureCanvas.height = webcamVideo.videoHeight;

        // Flip the canvas horizontally so the final photo is oriented correctly
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

uploadBtn.addEventListener('click', () => {
    // Stop camera and immediately clear the result image so the box is empty
    stopCamera(); 
    resultImage.src = "";
    resultImage.classList.add('hidden');
    fileInput.value = ""; 
    fileInput.click();
});

fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    await uploadAndProcess(file);
});

async function uploadAndProcess(file) {
    // Show a preview of the original image immediately under the loading screen
    const reader = new FileReader();
    reader.onload = (e) => {
        resultImage.src = e.target.result;
        resultImage.classList.remove('hidden');
        resultImage.style.opacity = "0.6"; // Preview is visible and clearer
    };
    reader.readAsDataURL(file);

    loading.classList.remove('hidden');
    
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('API Error');

        const blob = await response.blob();
        const imageUrl = URL.createObjectURL(blob);
        
        // Replace preview with final result and restore opacity
        resultImage.src = imageUrl;
        resultImage.style.opacity = opacitySlider.value / 100;
    } catch (err) {
        alert("Error connecting to AI: " + err.message);
        resultImage.style.opacity = "1";
    } finally {
        loading.classList.add('hidden');
    }
}

// Adjust the visibility of the result image based on the slider value
opacitySlider.addEventListener('input', (e) => {
    resultImage.style.opacity = e.target.value / 100;
});

// Clear the display and stop the camera when resetting the view
document.getElementById('resetView').addEventListener('click', () => {
    stopCamera();
    resultImage.src = "";
    resultImage.classList.add('hidden');
    fileInput.value = "";
});