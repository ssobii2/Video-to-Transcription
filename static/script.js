// Global variables
let systemInfo = null;
let availableModels = [];
let installedModels = [];
let isProcessing = false;

// Loading state management
const LoadingManager = {
  show: function (text = "Loading...", subtext = "") {
    this.hide(); // Remove any existing overlay

    const overlay = document.createElement("div");
    overlay.className = "loading-overlay";
    overlay.id = "loading-overlay";

    const content = document.createElement("div");
    content.className = "loading-content";

    const spinner = document.createElement("div");
    spinner.className = "loading-spinner";

    const textDiv = document.createElement("div");
    textDiv.className = "loading-text";
    textDiv.textContent = text;

    content.appendChild(spinner);
    content.appendChild(textDiv);

    if (subtext) {
      const subtextDiv = document.createElement("div");
      subtextDiv.className = "loading-subtext";
      subtextDiv.textContent = subtext;
      content.appendChild(subtextDiv);
    }

    overlay.appendChild(content);
    document.body.appendChild(overlay);
  },

  hide: function () {
    const existing = document.getElementById("loading-overlay");
    if (existing) {
      existing.remove();
    }
  },

  updateText: function (text, subtext = "") {
    const textDiv = document.querySelector(".loading-text");
    const subtextDiv = document.querySelector(".loading-subtext");

    if (textDiv) textDiv.textContent = text;

    if (subtextDiv && subtext) {
      subtextDiv.textContent = subtext;
    } else if (subtext && !subtextDiv) {
      const newSubtextDiv = document.createElement("div");
      newSubtextDiv.className = "loading-subtext";
      newSubtextDiv.textContent = subtext;
      document.querySelector(".loading-content").appendChild(newSubtextDiv);
    }
  },

  showButtonLoading: function (buttonId, originalText) {
    const button = document.getElementById(buttonId);
    if (button) {
      button.classList.add("loading");
      button.disabled = true;
      button.dataset.originalText = originalText || button.textContent;
    }
  },

  hideButtonLoading: function (buttonId, restoreText = true) {
    const button = document.getElementById(buttonId);
    if (button) {
      button.classList.remove("loading");
      button.disabled = false;
      if (restoreText && button.dataset.originalText) {
        button.textContent = button.dataset.originalText;
        delete button.dataset.originalText;
      }
    }
  },
};

// Initialize the application
document.addEventListener("DOMContentLoaded", function () {
  initializeApp();
});

async function initializeApp() {
  LoadingManager.show(
    "Initializing application...",
    "Loading system information and models"
  );

  try {
    await loadSystemInfo();
    LoadingManager.updateText(
      "Loading models...",
      "Detecting available models"
    );
    await loadModels();
    LoadingManager.updateText(
      "Loading AI prompts...",
      "Setting up AI features"
    );
    await loadSuggestedPrompts();
    LoadingManager.updateText("Setting up interface...", "Almost ready");
    setupEventListeners();
    listTranscriptions();
    listAIResponses();

    // Setup WebSocket for real-time updates
    setupWebSocket();
    setupDownloadWebSocket();

    LoadingManager.hide();
  } catch (error) {
    LoadingManager.hide();
    displayError("Failed to initialize application: " + error.message);
    console.error("Initialization error:", error);
  }
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

  let infoText = "SYSTEM";

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
  modelSelect.innerHTML =
    '<option value="" disabled>Select a model...</option>';

  availableModels.forEach((model) => {
    const option = document.createElement("option");
    option.value = model.size;
    
    if (model.description.toLowerCase().includes("recommended")) {
      option.textContent = `${model.name} - ${model.description}`;
    } else {
      option.textContent = `${model.name} - ${model.description}`;
      if (model.recommended) {
        option.textContent += " (Recommended)";
      }
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

  const file = fileInput.files[0];
  const prompt = promptInput.value.trim();
  const model = modelSelect.value;

  clearError();
  clearProgress();

  // Check for duplicate transcription before starting upload
  try {
    const duplicateCheckData = new FormData();
    duplicateCheckData.append("filename", file.name);
    
    const duplicateResponse = await fetch("/api/check-duplicate", {
      method: "POST",
      body: duplicateCheckData
    });

    if (duplicateResponse.ok) {
      const duplicateResult = await duplicateResponse.json();
      
      if (duplicateResult.exists) {
        updateMessage(`ℹ️ File Already Processed - ${duplicateResult.message}. You can find the existing files in the sections below.`);
        return;
      }
    } else {
      // If duplicate check fails, show warning but allow upload to proceed
      console.warn("Duplicate check failed, proceeding with upload");
    }
  } catch (error) {
    // If duplicate check fails, show warning but allow upload to proceed
    console.warn("Duplicate check error:", error);
  }

  // Show button loading state
  LoadingManager.showButtonLoading("upload-button", "Upload & Process");
  document.getElementById("status").style.display = "block";

  // Show progress section with spinner immediately
  const progressContainer = document.getElementById("progress-container");
  const progressSpinner = document.getElementById("progress-spinner");
  const messageElement = document.getElementById("message");

  progressContainer.style.display = "block";
  progressSpinner.style.display = "block";
  messageElement.innerText = "Uploading file...";

  uploadProgressBar.style.display = "block";
  uploadProgressMessage.style.display = "block";
  document.getElementById("upload-progress-container").style.display = "block";

  uploadProgressBar.value = 0;
  uploadProgressMessage.textContent = "";
  isProcessing = true;

  try {
    // Use chunked upload for large files (> 50MB) to bypass Cloudflare limits
    const CHUNK_SIZE = 50 * 1024 * 1024; // 50MB chunks to stay under Cloudflare's 100MB limit
    
    if (file.size > CHUNK_SIZE) {
      await handleChunkedUpload(file, prompt, model, uploadProgressBar, uploadProgressMessage);
    } else {
      await handleRegularUpload(file, prompt, model, uploadProgressBar, uploadProgressMessage);
    }

  } catch (error) {
    LoadingManager.hideButtonLoading("upload-button");
    isProcessing = false;
    displayError("Error uploading file: " + error.message);
    document.getElementById("status").style.display = "none";

    uploadProgressBar.style.display = "none";
    uploadProgressMessage.style.display = "none";
    document.getElementById("upload-progress-container").style.display = "none";
  }
}

async function handleRegularUpload(file, prompt, model, uploadProgressBar, uploadProgressMessage) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("prompt", prompt);
  formData.append("model", model);

  return new Promise((resolve, reject) => {
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
          hideUploadProgress();
          showProcessingStatus();
        }
      }
    };

    xhr.onload = function () {
      LoadingManager.hideButtonLoading("upload-button");
      resetFormState();
      
      if (xhr.status === 200) {
        try {
          const responseData = JSON.parse(xhr.responseText);
          if (responseData.error) {
            displayError(responseData.error);
          } else {
            listTranscriptions();
            listAIResponses();
          }
          resolve();
        } catch (e) {
          console.error("Error parsing response:", e);
          displayError("Unexpected response format from server");
          reject(e);
        }
      } else {
        handleUploadError(xhr);
        reject(new Error(`Upload failed with status ${xhr.status}`));
      }
    };

    xhr.onerror = function () {
      LoadingManager.hideButtonLoading("upload-button");
      isProcessing = false;
      displayError("Error uploading file.");
      document.getElementById("status").style.display = "none";
      hideUploadProgress();
      reject(new Error("Network error"));
    };

    xhr.send(formData);
  });
}

async function handleChunkedUpload(file, prompt, model, uploadProgressBar, uploadProgressMessage) {
  const CHUNK_SIZE = 50 * 1024 * 1024; // 50MB chunks
  const totalChunks = Math.ceil(file.size / CHUNK_SIZE);
  const fileName = file.name;
  
  uploadProgressMessage.textContent = `Upload Progress: 0%`;

  // Step 1: Initialize chunked upload
  try {
    const initResponse = await fetch("/upload/init", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        filename: fileName,
        filesize: file.size,
        total_chunks: totalChunks,
        prompt: prompt,
        model: model
      })
    });

    if (!initResponse.ok) {
      throw new Error(`Failed to initialize upload: ${initResponse.status}`);
    }

    const { upload_id } = await initResponse.json();

    // Step 2: Upload chunks with real-time progress
    let totalBytesUploaded = 0;
    
    for (let chunkIndex = 0; chunkIndex < totalChunks; chunkIndex++) {
      const start = chunkIndex * CHUNK_SIZE;
      const end = Math.min(start + CHUNK_SIZE, file.size);
      const chunk = file.slice(start, end);

      const chunkFormData = new FormData();
      chunkFormData.append("chunk", chunk);
      chunkFormData.append("upload_id", upload_id);
      chunkFormData.append("chunk_index", chunkIndex);
      chunkFormData.append("total_chunks", totalChunks);

      // Use XMLHttpRequest for this chunk to get real-time progress
      await new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open("POST", "/upload/chunk", true);

        xhr.upload.onprogress = function (event) {
          if (event.lengthComputable) {
            // Calculate total progress including previous chunks + current chunk progress
            const currentChunkProgress = event.loaded;
            const overallProgress = ((totalBytesUploaded + currentChunkProgress) / file.size) * 100;
            
            uploadProgressBar.value = overallProgress;
            uploadProgressMessage.textContent = `Upload Progress: ${Math.round(overallProgress)}%`;
          }
        };

        xhr.onload = function () {
          if (xhr.status === 200) {
            // Update total bytes uploaded after chunk completes
            totalBytesUploaded += chunk.size;
            resolve();
          } else {
            reject(new Error(`Failed to upload chunk ${chunkIndex + 1}: ${xhr.status}`));
          }
        };

        xhr.onerror = function () {
          reject(new Error(`Network error uploading chunk ${chunkIndex + 1}`));
        };

        xhr.send(chunkFormData);
      });
    }

    // Step 3: Complete upload
    const completeResponse = await fetch("/upload/complete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ upload_id: upload_id })
    });

    if (!completeResponse.ok) {
      throw new Error(`Failed to complete upload: ${completeResponse.status}`);
    }

    uploadProgressMessage.textContent = "Upload Complete!";
    hideUploadProgress();
    showProcessingStatus();

    LoadingManager.hideButtonLoading("upload-button");
    resetFormState();

    const responseData = await completeResponse.json();
    if (responseData.error) {
      displayError(responseData.error);
    } else {
      listTranscriptions();
      listAIResponses();
    }

  } catch (error) {
    throw error;
  }
}

function hideUploadProgress() {
  const uploadProgressBar = document.getElementById("upload-progress-bar");
  const uploadProgressMessage = document.getElementById("upload-progress-message");
  
  uploadProgressBar.style.display = "none";
  uploadProgressMessage.style.display = "none";
  document.getElementById("upload-progress-container").style.display = "none";
}

function showProcessingStatus() {
  document.getElementById("status").style.display = "block";
  
  // Update progress section for processing with a slight delay
  setTimeout(() => {
    const messageElement = document.getElementById("message");
    if (messageElement.innerText === "Uploading file...") {
      messageElement.innerText = "Upload complete, starting processing...";
    }
  }, 500);
}

function resetFormState() {
  document.getElementById("status").style.display = "none";
  document.getElementById("file-input").value = "";
  document.getElementById("selected-file-name").textContent = "No file selected";
  document.getElementById("selected-file-name").classList.remove("has-file");
  document.getElementById("prompt-input").value = "";
  isProcessing = false;
}

function handleUploadError(xhr) {
  // Handle error responses - check if response is JSON or HTML
  let errorMessage = "Error uploading file.";
  
  try {
    const responseData = JSON.parse(xhr.responseText);
    
    // Handle specific HTTP error codes
    if (
      xhr.status === 400 &&
      responseData.detail &&
      (responseData.detail.includes("already exists") ||
        responseData.detail.includes("Transcription already exists"))
    ) {
      errorMessage = `⚠️ File Already Processed\n\nA transcription already exists for this file. You can find it in the "Transcription Files" section below.\n\nIf you want to create a new transcription, please delete the existing one first.`;
    } else {
      errorMessage = responseData.detail || responseData.error || `Server error (${xhr.status})`;
    }
  } catch (e) {
    // Response is not JSON (likely HTML error page from nginx)
    console.error("Non-JSON response received:", xhr.responseText);
    errorMessage = `Server Error (${xhr.status})\n\nReceived an unexpected response from the server. Please try again or contact support if the issue persists.`;
  }
  
  displayError(errorMessage);
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
    // Show loading overlay
    LoadingManager.show(
      "Preparing download...",
      `Setting up download for ${modelName} model`
    );

    // Disable the download button to prevent multiple downloads
    const downloadButtons = document.querySelectorAll(
      `button[onclick="downloadModel('${modelName}')"]`
    );
    downloadButtons.forEach((btn) => {
      btn.classList.add("loading");
      btn.disabled = true;
    });

    updateMessage(`📥 Starting download for ${modelName} model...`);

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
      LoadingManager.updateText(
        "Download started",
        "Progress updates will appear above"
      );
              updateMessage(`${data.message} - Watch for progress updates above.`);

      // Keep the button disabled and show downloading state
      downloadButtons.forEach((btn) => {
        btn.classList.remove("loading");
        btn.textContent = "Downloading...";
        btn.style.background = "#17a2b8";
      });

      // Hide loading overlay after a short delay since WebSocket will handle progress
      setTimeout(() => LoadingManager.hide(), 2000);
    } else {
      const errorData = await response.json();
      LoadingManager.hide();

      // Handle turbo model error specifically
      if (errorData.detail && errorData.detail.includes("turbo")) {
        updateMessage(`❌ Model Not Supported - ${errorData.detail}`);
      } else {
        updateMessage(`❌ Error downloading model: ${errorData.detail || "Unknown error"}`);
      }

      // Re-enable download buttons on error
      downloadButtons.forEach((btn) => {
        btn.classList.remove("loading");
        btn.disabled = false;
        btn.textContent = "Download";
        btn.style.background = "#28a745";
      });
    }
  } catch (error) {
    LoadingManager.hide();
    console.error("Error downloading model:", error);
    updateMessage("❌ Error downloading model");

    // Re-enable download buttons on error
    const downloadButtons = document.querySelectorAll(
      `button[onclick="downloadModel('${modelName}')"]`
    );
    downloadButtons.forEach((btn) => {
      btn.classList.remove("loading");
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

        // Hide any loading overlays
        LoadingManager.hide();

        // Re-enable and update download buttons
        setTimeout(() => {
          const downloadButtons = document.querySelectorAll(
            `button[onclick="downloadModel('${modelName}')"]`
          );
          downloadButtons.forEach((btn) => {
            btn.classList.remove("loading");
            btn.disabled = false;
            btn.textContent = "Download";
            btn.style.background = "#28a745";
          });

          // Refresh models list to show the newly downloaded model
          loadModels();
          updateMessage(`✅ ${modelName} model is now ready for use!`);

          // Auto-hide status after 3 seconds
          setTimeout(() => {
            const progressContainer =
              document.getElementById("progress-container");
            if (progressContainer) {
              progressContainer.style.display = "none";
            }
          }, 3000);
        }, 2000);
      }
    }

    // Check if this is a download failure message
    if (
      message.includes("Failed to download") ||
      message.includes("was cancelled")
    ) {
      // Hide any loading overlays
      LoadingManager.hide();

      // Extract model name and re-enable buttons
      const modelMatch = message.match(/(\w+) model/);
      if (modelMatch) {
        const modelName = modelMatch[1];
        const downloadButtons = document.querySelectorAll(
          `button[onclick="downloadModel('${modelName}')"]`
        );
        downloadButtons.forEach((btn) => {
          btn.classList.remove("loading");
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
    LoadingManager.show(
      "Deleting model...",
      `Removing ${modelName} model from storage`
    );
    updateMessage(`Deleting ${modelName} model...`);

    const response = await fetch(`/api/models/${modelName}`, {
      method: "DELETE",
    });

    if (response.ok) {
      LoadingManager.updateText("Model deleted", "Refreshing model list");
              updateMessage(`${modelName} model deleted successfully!`);
      // Refresh models list
      await loadModels();
      LoadingManager.hide();

      // Auto-hide status after 3 seconds
      setTimeout(() => {
        const progressContainer = document.getElementById("progress-container");
        if (progressContainer) {
          progressContainer.style.display = "none";
        }
      }, 3000);
    } else {
      LoadingManager.hide();
      const errorData = await response.json();
      updateMessage(`❌ Error deleting model: ${errorData.detail || "Unknown error"}`);
    }
  } catch (error) {
    LoadingManager.hide();
    console.error("Error deleting model:", error);
    updateMessage("❌ Error deleting model");
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
  const errorDiv = document.getElementById("error-message");
  errorDiv.textContent = message;
  errorDiv.style.display = "block";
  errorDiv.style.backgroundColor = "#f8d7da";
  errorDiv.style.color = "#721c24";
  errorDiv.style.border = "1px solid #f5c6cb";
}

function displayMessage(message) {
  const errorDiv = document.getElementById("error-message");
  errorDiv.textContent = message;
  errorDiv.style.display = "block";
  errorDiv.style.backgroundColor = "#d1ecf1";
  errorDiv.style.color = "#0c5460";
  errorDiv.style.border = "1px solid #bee5eb";
}

function clearError() {
  const errorDiv = document.getElementById("error-message");
  errorDiv.textContent = "";
  errorDiv.style.display = "none";
}

function clearProgress() {
  const messageElement = document.getElementById("message");
  const progressContainer = document.getElementById("progress-container");
  const progressSpinner = document.getElementById("progress-spinner");

  if (messageElement) {
    messageElement.innerText = "";
  }

  // Hide progress container and spinner
  progressContainer.style.display = "none";
  progressSpinner.style.display = "none";
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
                    <span title="${filename}">${filename}</span>
                    <div class="file-actions">
                        <button onclick="downloadTranscription('${filename}')">Download</button>
                        <button onclick="deleteTranscription('${filename}')">Delete</button>
                    </div>
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
                    <span title="${filename}">${filename}</span>
                    <div class="file-actions">
                        <button onclick="downloadAIResponse('${filename}')">Download</button>
                        <button onclick="deleteAIResponse('${filename}')">Delete</button>
                    </div>
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
    LoadingManager.show("Deleting file...", `Removing ${filename}`);

    const response = await fetch(`/transcription/${filename}`, {
      method: "DELETE",
    });

    if (response.ok) {
      LoadingManager.hide();
      listTranscriptions();
      updateMessage(`${filename} deleted successfully`);
    } else {
      LoadingManager.hide();
      updateMessage("❌ Error deleting transcription.");
    }
  } catch (error) {
    LoadingManager.hide();
    updateMessage("❌ Error deleting transcription.");
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
    LoadingManager.show("Deleting file...", `Removing ${filename}`);

    const response = await fetch(`/ai/${filename}`, {
      method: "DELETE",
    });

    if (response.ok) {
      LoadingManager.hide();
      listAIResponses();
      updateMessage(`${filename} deleted successfully`);
    } else {
      LoadingManager.hide();
      updateMessage("❌ Error deleting AI response.");
    }
  } catch (error) {
    LoadingManager.hide();
    updateMessage("❌ Error deleting AI response.");
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
  const progressSpinner = document.getElementById("progress-spinner");

  if (messageElement) {
    // Don't override upload completion message too quickly for audio files
    const currentMessage = messageElement.innerText;
    if (
      currentMessage.includes("Upload complete, starting processing...") &&
      message.includes("Preparing audio file...")
    ) {
      // Give the user a moment to see the upload completion
      setTimeout(() => {
        if (
          messageElement.innerText.includes(
            "Upload complete, starting processing..."
          )
        ) {
          messageElement.innerText = message;
        }
      }, 1500);
      return; // Don't continue with the rest of the function yet
    } else {
      messageElement.innerText = message;
    }
  }

  // Show progress container when we have a message, keep it visible during processing
  if (message && message.trim()) {
    progressContainer.style.display = "block";

    // Show spinner for processing messages, hide for completion/error messages
    if (
      message.includes("Processing ") ||
      message.includes("Starting transcription...") ||
      message.includes("Processing with AI...") ||
      message.includes("Formatting transcription...") ||
      message.includes("Saving transcription...") ||
      message.includes("Converting") ||
      message.includes("Extracting") ||
      message.includes("Preparing audio file...") ||
      message.includes("Audio file ready")
    ) {
      // Show spinner during active processing
      progressSpinner.style.display = "block";
    } else if (
      message.includes("completed successfully") ||
      message.includes("❌") ||
      message.includes("⚠️") ||
      message.includes("ℹ️") ||
      message.includes("Processing failed") ||
      message.includes("skipped")
    ) {
      // Hide spinner when processing ends (success, error, or skip)
      progressSpinner.style.display = "none";
    }

    // Only hide progress container when processing is completely done
    if (
      message.includes("Processing completed successfully") ||
      message.includes("❌") ||
      message.includes("Processing failed") ||
      message.includes("deleted successfully") ||
      message.includes("is now ready for use") ||
      message.includes("ℹ️ File Already Processed")
    ) {
      // Keep it visible for a few seconds so user can see completion message
      setTimeout(() => {
        progressContainer.style.display = "none";
        progressSpinner.style.display = "none";
      }, 4000);
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
