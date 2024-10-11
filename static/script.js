document.getElementById("upload-button").onclick = async function() {
    const fileInput = document.getElementById("file-input");
    if (fileInput.files.length === 0) {
        displayError("Please select a file to upload.");
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/upload/");
    
    xhr.onload = function() {
        const responseData = JSON.parse(xhr.responseText);
        document.getElementById("status").style.display = "none";
        
        if (xhr.status === 200) {
            if (responseData.error) {
                displayError(responseData.error);
                document.getElementById("upload-button").disabled = false;
                document.getElementById("file-input").value = "";  // Reset file input
            } else {
                listTranscriptions();
                document.getElementById("file-input").value = "";  // Reset file input
            }
        } else {
            displayError(responseData.error || "Error uploading file.");
            document.getElementById("upload-button").disabled = false;
        }
    };

    xhr.onerror = function() {
        displayError("Error uploading file.");
        document.getElementById("status").style.display = "none";
        document.getElementById("upload-button").disabled = false;
    };

    xhr.send(formData);
    document.getElementById("status").style.display = "block";
    clearError();
    document.getElementById("upload-button").disabled = true;
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

async function listTranscriptions() {
    const response = await fetch("/transcription/");
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
    const response = await fetch(`/transcription/${filename}`, {
        method: 'DELETE',
    });

    if (response.ok) {
        listTranscriptions();
    } else {
        const errorData = await response.json();
        displayError(errorData.error || "Error deleting transcription.");
    }
}

window.onload = listTranscriptions;