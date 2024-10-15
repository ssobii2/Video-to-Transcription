document.getElementById("upload-button").onclick = async function() {
    const fileInput = document.getElementById("file-input");
    const promptInput = document.getElementById("prompt-input");
    
    if (fileInput.files.length === 0) {
        displayError("Please select a file to upload.");
        return;
    }
    
    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    formData.append("prompt", promptInput.value);
    
    clearError();
    clearProgress();
    document.getElementById("status").style.display = "block";
    document.getElementById("upload-button").disabled = true;

    try {
        const response = await fetch("/upload/", {
            method: "POST",
            body: formData,
        });

        document.getElementById("status").style.display = "none";
        document.getElementById("file-input").value = "";
        document.getElementById("upload-button").disabled = false;

        if (response.ok) {
            const responseData = await response.json();
            if (responseData.error) {
                displayError(responseData.error);
            } else {
                listTranscriptions();
            }
        } else {
            const responseData = await response.json();
            displayError(responseData.error || "Error uploading file.");
        }
    } catch (error) {
        displayError("Error uploading file.");
        document.getElementById("status").style.display = "none";
        document.getElementById("upload-button").disabled = false;
    }
};

function displayError(message) {
    const errorMessageElement = document.getElementById("error-message");
    errorMessageElement.innerText = message;
    errorMessageElement.style.display = "block";
}

function clearError() {
    const errorMessageElement = document.getElementById("error-message");
    errorMessageElement.innerText = "";
    errorMessageElement.style.display = "none";
}

function clearProgress() {
    const messageElement = document.getElementById("message");
    if (messageElement) {
        messageElement.innerText = "";
    }
}

async function listTranscriptions() {
    try {
        const response = await fetch("/transcription/");
        if (response.ok) {
            const { transcriptions } = await response.json();

            const transcriptionContainer = document.getElementById("transcriptions");
            transcriptionContainer.innerHTML = '';
            transcriptions.forEach(filename => {
                const item = document.createElement('div');
                item.className = 'transcription-item';
                item.innerHTML = `
                    <span>${filename}</span>
                    <button onclick="downloadTranscription('${filename}')">Download</button>
                    <button onclick="deleteTranscription('${filename}')">Delete</button>
                `;
                transcriptionContainer.appendChild(item);
            });
        } else {
            displayError("Error fetching transcriptions.");
        }
    } catch (error) {
        displayError("Error fetching transcriptions.");
    }
}

function downloadTranscription(filename) {
    const link = document.createElement('a');
    link.href = `/transcription/${filename}`;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

async function deleteTranscription(filename) {
    try {
        const response = await fetch(`/transcription/${filename}`, {
            method: 'DELETE',
        });

        if (response.ok) {
            listTranscriptions();
        } else {
            const errorData = await response.json();
            displayError(errorData.error || "Error deleting transcription.");
        }
    } catch (error) {
        displayError("Error deleting transcription.");
    }
}

// Establish WebSocket connection
const socket = new WebSocket(`ws://${window.location.host}/ws`);

socket.onmessage = function(event) {
    const message = event.data;
    updateProgress(message);
};

socket.onopen = function() {
    console.log("WebSocket connection established.");
};

socket.onclose = function(event) {
    console.log("WebSocket connection closed.");
};

function updateProgress(message) {
    const messageElement = document.getElementById("message");
    messageElement.innerText = message;
}

window.onload = listTranscriptions;
