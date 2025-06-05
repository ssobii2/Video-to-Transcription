// Global variables
let systemInfo = null;
let availableModels = [];
let installedModels = [];
let isProcessing = false;

// Initialize the application
document.addEventListener("DOMContentLoaded", function () {
  initializeApp();
});

async function initializeApp() {
  await loadSystemInfo();
  await loadModels();
  await loadSuggestedPrompts();
  setupEventListeners();
  listTranscriptions();
  listAIResponses();

  // Setup WebSocket for real-time updates
  setupWebSocket();
  setupDownloadWebSocket();
}

function setupEventListeners() {
  // File input change
  document.getElementById("file-input").addEventListener("change", function () {
    const fileInput = document.getElementById("file-input");
    const selectedFileName = document.getElementById("selected-file-name");
    if (fileInput.files.length > 0) {
      const fileName = truncateFileName(fileInput.files[0].name, 40);
      selectedFileName.textContent = fileName;
      selectedFileName.classList.add("has-file");
      enableUploadIfReady();
    } else {
      selectedFileName.textContent = "No file selected";
      selectedFileName.classList.remove("has-file");
      document.getElementById("upload-button").disabled = true;
    }
  });

  // Model selection change
  document
    .getElementById("model-select")
    .addEventListener("change", function () {
      updateModelInfo();
      enableUploadIfReady();
    });

  // Upload button
  document.getElementById("upload-button").onclick = handleUpload;

  // Suggested prompts button click
  document
    .getElementById("show-suggestions-btn")
    .addEventListener("click", function () {
      toggleSuggestedPrompts();
    });

  // Prompt suggestions change
  document
    .getElementById("prompt-suggestions")
    .addEventListener("change", function () {
      const selectedPrompt = this.value;
      if (selectedPrompt) {
        document.getElementById("prompt-input").value = selectedPrompt;
        // Hide the dropdown after selection
        document.getElementById("prompt-suggestions").style.display = "none";
        document.getElementById("show-suggestions-btn").style.display =
          "inline-block";
      }
    });
}

async function loadSuggestedPrompts() {
  try {
    const response = await fetch("/api/prompts");
    if (response.ok) {
      const data = await response.json();
      populateSuggestedPrompts(data.prompts);
      // Show the suggested prompts button if AI is available
      document.getElementById("show-suggestions-btn").style.display =
        "inline-block";
    } else {
      // AI service not available, hide the suggested prompts feature
      document.getElementById("show-suggestions-btn").style.display = "none";
    }
  } catch (error) {
    console.error("Error loading suggested prompts:", error);
    document.getElementById("show-suggestions-btn").style.display = "none";
  }
}

function populateSuggestedPrompts(prompts) {
  const promptSelect = document.getElementById("prompt-suggestions");

  // Clear existing options except the first one
  promptSelect.innerHTML =
    '<option value="">Select a suggested prompt...</option>';

  prompts.forEach((prompt) => {
    const option = document.createElement("option");
    option.value = prompt.prompt;
    option.textContent = `${prompt.name} - ${prompt.description}`;
    promptSelect.appendChild(option);
  });
}

function toggleSuggestedPrompts() {
  const promptSelect = document.getElementById("prompt-suggestions");
  const showButton = document.getElementById("show-suggestions-btn");

  if (
    promptSelect.style.display === "none" ||
    promptSelect.style.display === ""
  ) {
    promptSelect.style.display = "inline-block";
    showButton.style.display = "none";
  } else {
    promptSelect.style.display = "none";
    showButton.style.display = "inline-block";
  }
}

function enableUploadIfReady() {
  const fileInput = document.getElementById("file-input");
  const modelSelect = document.getElementById("model-select");
  const uploadButton = document.getElementById("upload-button");

  if (fileInput.files.length > 0 && modelSelect.value) {
    uploadButton.disabled = false;
  } else {
    uploadButton.disabled = true;
  }
}

async function loadSystemInfo() {
  try {
    const response = await fetch("/api/status");
    if (response.ok) {
      systemInfo = await response.json();
      updateSystemInfoDisplay();
    }
  } catch (error) {
    console.error("Error loading system info:", error);
  }
}

function updateSystemInfoDisplay() {
  if (!systemInfo) return;

  const hardwareInfo = document.getElementById("hardware-info");
  const hardware = systemInfo.model_info.hardware;

  let infoText = `${systemInfo.environment.toUpperCase()}`;

  if (hardware.has_gpu) {
    infoText += ` | GPU: ${hardware.gpu_name} (${hardware.gpu_memory_gb.toFixed(
      1
    )}GB)`;
  } else {
    infoText += ` | CPU: ${hardware.cpu_cores} cores`;
  }

  infoText += ` | RAM: ${hardware.ram_gb.toFixed(1)}GB`;

  hardwareInfo.textContent = infoText;
}

async function loadModels() {
  try {
    const response = await fetch("/api/models");
    if (response.ok) {
      const data = await response.json();
      availableModels = data.models;
      installedModels = data.installed_models || [];
      updateModelSelect();
      loadInstalledModelsList();
      loadCompatibleModelsList();
    }
  } catch (error) {
    console.error("Error loading models:", error);
    displayError("Error loading model information");
  }
}

function updateModelSelect() {
  const modelSelect = document.getElementById("model-select");
  modelSelect.innerHTML = '<option value="">Select a model...</option>';

  availableModels.forEach((model) => {
    const option = document.createElement("option");
    option.value = model.size;
    option.textContent = `${model.name} - ${model.description}`;

    if (model.recommended) {
      option.textContent += " (Recommended)";
    }

    // Disable if not installed
    if (!installedModels.includes(model.size)) {
      option.textContent += " (Not Downloaded)";
      option.disabled = true;
    }

    modelSelect.appendChild(option);
  });

  // Auto-select the first available installed model
  if (installedModels.length > 0) {
    const firstInstalled = availableModels.find((m) =>
      installedModels.includes(m.size)
    );
    if (firstInstalled) {
      modelSelect.value = firstInstalled.size;
      updateModelInfo();
    }
  }
}

function updateModelInfo() {
  const modelSelect = document.getElementById("model-select");
  const modelInfo = document.getElementById("model-info");

  if (!modelSelect.value) {
    modelInfo.textContent = "";
    return;
  }

  const selectedModel = availableModels.find(
    (m) => m.size === modelSelect.value
  );
  if (selectedModel) {
    modelInfo.textContent = `Selected: ${selectedModel.name} - ${selectedModel.description}`;
  }
}

async function handleUpload() {
  const fileInput = document.getElementById("file-input");
  const promptInput = document.getElementById("prompt-input");
  const modelSelect = document.getElementById("model-select");
  const uploadProgressBar = document.getElementById("upload-progress-bar");
  const uploadProgressMessage = document.getElementById(
    "upload-progress-message"
  );

  if (fileInput.files.length === 0) {
    displayError("Please select a file to upload.");
    return;
  }

  if (!modelSelect.value) {
    displayError("Please select a transcription model.");
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);
  const prompt = promptInput.value.trim();
  formData.append("prompt", prompt);
  formData.append("model", modelSelect.value);

  clearError();
  clearProgress();
  document.getElementById("status").style.display = "block";
  document.getElementById("upload-button").disabled = true;

  uploadProgressBar.style.display = "block";
  uploadProgressMessage.style.display = "block";
  document.getElementById("upload-progress-container").style.display = "block";

  uploadProgressBar.value = 0;
  uploadProgressMessage.textContent = "";
  isProcessing = true;

  try {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/upload/", true);

    xhr.upload.onprogress = function (event) {
      if (event.lengthComputable) {
        const percentComplete = (event.loaded / event.total) * 100;
        uploadProgressBar.value = percentComplete;
        uploadProgressMessage.textContent = `Upload Progress: ${Math.round(
          percentComplete
        )}%`;

        if (percentComplete >= 100) {
          uploadProgressMessage.textContent = "Upload Complete!";

          uploadProgressBar.style.display = "none";
          uploadProgressMessage.style.display = "none";
          document.getElementById("upload-progress-container").style.display =
            "none";

          document.getElementById("status").style.display = "block";
        }
      }
    };

    xhr.onload = async function () {
      if (xhr.status !== 200) {
        isProcessing = false;
      }
      document.getElementById("status").style.display = "none";
      document.getElementById("file-input").value = "";
      document.getElementById("selected-file-name").textContent =
        "No file selected";
      document
        .getElementById("selected-file-name")
        .classList.remove("has-file");
      document.getElementById("prompt-input").value = "";
      document.getElementById("upload-button").disabled = true;

      uploadProgressBar.style.display = "none";
      uploadProgressMessage.style.display = "none";
      document.getElementById("upload-progress-container").style.display =
        "none";

      if (xhr.status === 200) {
        const responseData = JSON.parse(xhr.responseText);
        if (responseData.error) {
          displayError(responseData.error);
        } else {
          listTranscriptions();
          listAIResponses();
        }
      } else {
        const responseData = JSON.parse(xhr.responseText);
        displayError(responseData.error || "Error uploading file.");
      }
    };

    xhr.onerror = function () {
      isProcessing = false;
      displayError("Error uploading file.");
      document.getElementById("status").style.display = "none";
      document.getElementById("upload-button").disabled = true;

      uploadProgressBar.style.display = "none";
      uploadProgressMessage.style.display = "none";
      document.getElementById("upload-progress-container").style.display =
        "none";
    };

    uploadProgressBar.style.display = "block";
    xhr.send(formData);
  } catch (error) {
    isProcessing = false;
    displayError("Error uploading file.");
    document.getElementById("status").style.display = "none";
    document.getElementById("upload-button").disabled = true;

    uploadProgressBar.style.display = "none";
    uploadProgressMessage.style.display = "none";
    document.getElementById("upload-progress-container").style.display = "none";
  }
}

// Model Management Functions
async function loadInstalledModelsList() {
  const container = document.getElementById("installed-models-list");

  if (installedModels.length === 0) {
    container.innerHTML =
      '<p>No models downloaded yet. Use the "Download Models" tab to download models.</p>';
    return;
  }

  container.innerHTML = "";

  installedModels.forEach((modelSize) => {
    const model = availableModels.find((m) => m.size === modelSize);
    if (model) {
      const modelItem = createModelItem(model, true);
      container.appendChild(modelItem);
    }
  });
}

async function loadCompatibleModelsList() {
  try {
    const response = await fetch("/api/models/compatible");
    if (response.ok) {
      const data = await response.json();
      displayCompatibleModels(data.compatible_models);
    }
  } catch (error) {
    console.error("Error loading compatible models:", error);
  }
}

function displayCompatibleModels(compatibleModels) {
  const container = document.getElementById("compatible-models-list");
  container.innerHTML = "";

  if (compatibleModels.length === 0) {
    container.innerHTML = "<p>No compatible models found.</p>";
    return;
  }

  compatibleModels.forEach((model) => {
    const isInstalled = installedModels.includes(model.size);
    const modelItem = createModelItem(model, isInstalled, true);
    container.appendChild(modelItem);
  });
}

function createModelItem(model, isInstalled, showDownload = false) {
  const item = document.createElement("div");
  item.className = `model-item ${model.recommended ? "recommended" : ""}`;

  let actionsHtml = "";
  if (isInstalled) {
    actionsHtml = `
      <span class="btn-small" style="background: #28a745; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px; margin-right: 8px;">Installed</span>
      <button class="btn-small btn-danger" onclick="deleteModel('${model.size}')" style="background: #dc3545; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px; border: none; cursor: pointer;">Delete</button>
    `;
  } else if (showDownload) {
    actionsHtml = `<button class="btn-small btn-success" onclick="downloadModel('${model.size}')">Download</button>`;
  }

  item.innerHTML = `
        <div class="model-info">
            <div class="model-name">${model.name}</div>
            <div class="model-description">${model.description}</div>
            <div class="model-size">${model.size} ${
    model.memory_req ? "| " + model.memory_req : ""
  }</div>
        </div>
        <div class="model-actions">
            ${actionsHtml}
        </div>
    `;

  return item;
}

async function downloadModel(modelName) {
  try {
    // Disable the download button to prevent multiple downloads
    const downloadButtons = document.querySelectorAll(
      `button[onclick="downloadModel('${modelName}')"]`
    );
    downloadButtons.forEach((btn) => {
      btn.disabled = true;
      btn.textContent = "Downloading...";
      btn.style.background = "#6c757d";
    });

    displayMessage(`📥 Starting download for ${modelName} model...`);

    // Setup WebSocket for real-time updates if not already connected
    if (!window.downloadWebSocket) {
      setupDownloadWebSocket();
    }

    const formData = new FormData();
    formData.append("whisper_model", modelName);

    const response = await fetch("/api/models/download", {
      method: "POST",
      body: formData,
    });

    if (response.ok) {
      const data = await response.json();
      displayMessage(`${data.message} - Watch for progress updates below.`);

      // Keep the button disabled and show downloading state
      downloadButtons.forEach((btn) => {
        btn.textContent = "Downloading...";
        btn.style.background = "#17a2b8";
      });
    } else {
      const errorData = await response.json();
      displayError(errorData.detail || "Error downloading model");

      // Re-enable download buttons on error
      downloadButtons.forEach((btn) => {
        btn.disabled = false;
        btn.textContent = "Download";
        btn.style.background = "#28a745";
      });
    }
  } catch (error) {
    console.error("Error downloading model:", error);
    displayError("Error downloading model");

    // Re-enable download buttons on error
    const downloadButtons = document.querySelectorAll(
      `button[onclick="downloadModel('${modelName}')"]`
    );
    downloadButtons.forEach((btn) => {
      btn.disabled = false;
      btn.textContent = "Download";
      btn.style.background = "#28a745";
    });
  }
}

function setupDownloadWebSocket() {
  if (
    window.downloadWebSocket &&
    window.downloadWebSocket.readyState === WebSocket.OPEN
  ) {
    return; // Already connected
  }

  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${location.host}/ws`;

  window.downloadWebSocket = new WebSocket(wsUrl);

  window.downloadWebSocket.onopen = function () {
    console.log("Download WebSocket connected");
  };

  window.downloadWebSocket.onmessage = function (event) {
    const message = event.data;

    // Check if this is a download completion message
    if (message.includes("downloaded successfully")) {
      // Extract model name from success message
      const modelMatch = message.match(/(\w+) model downloaded successfully/);
      if (modelMatch) {
        const modelName = modelMatch[1];

        // Re-enable and update download buttons
        setTimeout(() => {
          const downloadButtons = document.querySelectorAll(
            `button[onclick="downloadModel('${modelName}')"]`
          );
          downloadButtons.forEach((btn) => {
            btn.disabled = false;
            btn.textContent = "Download";
            btn.style.background = "#28a745";
          });

          // Refresh models list to show the newly downloaded model
          loadModels();
          displayMessage(`✅ ${modelName} model is now ready for use!`);
        }, 2000);
      }
    }

    // Check if this is a download failure message
    if (
      message.includes("Failed to download") ||
      message.includes("was cancelled")
    ) {
      // Extract model name and re-enable buttons
      const modelMatch = message.match(/(\w+) model/);
      if (modelMatch) {
        const modelName = modelMatch[1];
        const downloadButtons = document.querySelectorAll(
          `button[onclick="downloadModel('${modelName}')"]`
        );
        downloadButtons.forEach((btn) => {
          btn.disabled = false;
          btn.textContent = "Download";
          btn.style.background = "#28a745";
        });
      }
    }

    // Display the message in the progress area
    updateMessage(message);
  };

  window.downloadWebSocket.onclose = function () {
    console.log("Download WebSocket disconnected");
    window.downloadWebSocket = null;

    // Reconnect after a delay if the page is still active
    setTimeout(() => {
      if (!window.downloadWebSocket && document.visibilityState === "visible") {
        setupDownloadWebSocket();
      }
    }, 3000);
  };

  window.downloadWebSocket.onerror = function (error) {
    console.error("Download WebSocket error:", error);
    window.downloadWebSocket = null;
  };
}

async function deleteModel(modelName) {
  // Show confirmation dialog
  if (
    !confirm(
      `Are you sure you want to delete the ${modelName} model? This will free up disk space but you'll need to download it again to use it.`
    )
  ) {
    return;
  }

  try {
    displayMessage(`Deleting ${modelName} model...`);

    const response = await fetch(`/api/models/${modelName}`, {
      method: "DELETE",
    });

    if (response.ok) {
      displayMessage(`${modelName} model deleted successfully!`);
      // Refresh models list
      await loadModels();
    } else {
      const errorData = await response.json();
      displayError(errorData.detail || "Error deleting model");
    }
  } catch (error) {
    console.error("Error deleting model:", error);
    displayError("Error deleting model");
  }
}

// Tab Management
function openTab(evt, tabName) {
  var i, tabcontent, tablinks;

  // Hide all tab content
  tabcontent = document.getElementsByClassName("tab-content");
  for (i = 0; i < tabcontent.length; i++) {
    tabcontent[i].classList.remove("active");
  }

  // Remove active class from all tab buttons
  tablinks = document.getElementsByClassName("tab-button");
  for (i = 0; i < tablinks.length; i++) {
    tablinks[i].classList.remove("active");
  }

  // Show the selected tab and mark button as active
  document.getElementById(tabName).classList.add("active");
  evt.currentTarget.classList.add("active");
}

// Utility Functions
function truncateFileName(fileName, maxLength) {
  if (fileName.length <= maxLength) {
    return fileName;
  }
  const extension = fileName.split(".").pop();
  const truncatedName = fileName.substring(0, maxLength - extension.length - 3);
  return `${truncatedName}...${extension}`;
}

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

function displayMessage(message) {
  const messageElement = document.getElementById("message");
  if (messageElement) {
    messageElement.innerText = message;
  }
}

function clearProgress() {
  const messageElement = document.getElementById("message");
  if (messageElement) {
    messageElement.innerText = "";
  }
}

// File Management Functions
async function listTranscriptions() {
  try {
    const response = await fetch("/transcription/");
    if (response.ok) {
      const { transcriptions } = await response.json();

      const transcriptionContainer = document.getElementById("transcriptions");
      transcriptionContainer.innerHTML = "";

      if (transcriptions.length === 0) {
        transcriptionContainer.innerHTML = "<p>No transcription files yet.</p>";
        return;
      }

      const sortedTranscriptions = transcriptions.sort((a, b) => {
        return b.localeCompare(a);
      });

      sortedTranscriptions.forEach((filename) => {
        const item = document.createElement("div");
        item.className = "transcription-item";
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

async function listAIResponses() {
  try {
    const response = await fetch("/ai/");
    if (response.ok) {
      const { ai_responses } = await response.json();

      const aiContainer = document.getElementById("ai-responses");
      aiContainer.innerHTML = "";

      if (ai_responses.length === 0) {
        aiContainer.innerHTML = "<p>No AI response files yet.</p>";
        return;
      }

      const sortedResponses = ai_responses.sort((a, b) => {
        return b.localeCompare(a);
      });

      sortedResponses.forEach((filename) => {
        const item = document.createElement("div");
        item.className = "ai-response-item";
        item.innerHTML = `
                    <span>${filename}</span>
                    <button onclick="downloadAIResponse('${filename}')">Download</button>
                    <button onclick="deleteAIResponse('${filename}')">Delete</button>
                `;
        aiContainer.appendChild(item);
      });
    } else {
      displayError("Error fetching AI responses.");
    }
  } catch (error) {
    displayError("Error fetching AI responses.");
  }
}

function downloadTranscription(filename) {
  const link = document.createElement("a");
  link.href = `/transcription/${filename}`;
  link.download = filename;
  link.click();
}

async function deleteTranscription(filename) {
  if (!confirm(`Are you sure you want to delete ${filename}?`)) {
    return;
  }

  try {
    const response = await fetch(`/transcription/${filename}`, {
      method: "DELETE",
    });

    if (response.ok) {
      listTranscriptions();
    } else {
      displayError("Error deleting transcription.");
    }
  } catch (error) {
    displayError("Error deleting transcription.");
  }
}

function downloadAIResponse(filename) {
  const link = document.createElement("a");
  link.href = `/ai/${filename}`;
  link.download = filename;
  link.click();
}

async function deleteAIResponse(filename) {
  if (!confirm(`Are you sure you want to delete ${filename}?`)) {
    return;
  }

  try {
    const response = await fetch(`/ai/${filename}`, {
      method: "DELETE",
    });

    if (response.ok) {
      listAIResponses();
    } else {
      displayError("Error deleting AI response.");
    }
  } catch (error) {
    displayError("Error deleting AI response.");
  }
}

// WebSocket for real-time updates
function setupWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws`;
  const ws = new WebSocket(wsUrl);

  ws.onmessage = function (event) {
    updateMessage(event.data);
  };

  ws.onclose = function () {
    // Reconnect after 3 seconds
    setTimeout(setupWebSocket, 3000);
  };

  ws.onerror = function (error) {
    console.error("WebSocket error:", error);
  };
}

function updateMessage(message) {
  const messageElement = document.getElementById("message");
  const progressContainer = document.getElementById("progress-container");

  if (messageElement) {
    messageElement.innerText = message;
  }

  // Show progress container when we have a message, keep it visible during processing
  if (message && message.trim()) {
    progressContainer.style.display = "block";

    // Only hide progress container when processing is completely done
    if (message.includes("Processing completed successfully")) {
      // Keep it visible for a few seconds so user can see completion message
      setTimeout(() => {
        progressContainer.style.display = "none";
      }, 3000);
    }
  }

  // Auto-refresh file lists when processing completes
  if (
    message.includes("Processing completed successfully") ||
    message.includes("Transcription saved successfully") ||
    message.includes("AI processing completed")
  ) {
    // Refresh file lists after a short delay to ensure files are written
    setTimeout(() => {
      listTranscriptions();
      listAIResponses();
    }, 1000);
  }
}

// Initialize WebSocket connection
setupWebSocket();
