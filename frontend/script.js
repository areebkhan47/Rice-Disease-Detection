// ==========================================
// ELEMENTS
// ==========================================

const imageInput = document.getElementById("imageInput");

const uploadArea = document.getElementById("uploadArea");

const browseBtn = document.getElementById("browseBtn");

const analyzeBtn = document.getElementById("analyzeBtn");

const buttonText = document.getElementById("buttonText");

const buttonLoader = document.getElementById("buttonLoader");

const uploadPlaceholder = document.getElementById("uploadPlaceholder");

const previewContainer = document.getElementById("previewContainer");

const imagePreview = document.getElementById("imagePreview");

const removeImage = document.getElementById("removeImage");

const errorMessage = document.getElementById("errorMessage");

const loadingSection = document.getElementById("loadingSection");

const resultsSection = document.getElementById("resultsSection");

const newAnalysisBtn = document.getElementById("newAnalysisBtn");

let selectedFile = null;


// ==========================================
// IMAGE SELECTION
// ==========================================

browseBtn.addEventListener("click", () => {
    imageInput.click();
});


imageInput.addEventListener("change", (event) => {

    const file = event.target.files[0];

    if (file) {
        handleImage(file);
    }

});


// ==========================================
// HANDLE IMAGE
// ==========================================

function handleImage(file) {

    hideError();

    const allowedTypes = [
        "image/jpeg",
        "image/png",
        "image/webp"
    ];

    if (!allowedTypes.includes(file.type)) {

        showError("Please upload a JPG, PNG or WEBP image.");

        return;
    }

    if (file.size > 8 * 1024 * 1024) {

        showError("Image size must be less than 8 MB.");

        return;
    }

    selectedFile = file;

    const reader = new FileReader();

    reader.onload = function(event) {

        imagePreview.src = event.target.result;

        uploadPlaceholder.classList.add("hidden");

        previewContainer.classList.remove("hidden");

        analyzeBtn.disabled = false;

    };

    reader.readAsDataURL(file);

}


// ==========================================
// REMOVE IMAGE
// ==========================================

removeImage.addEventListener("click", () => {

    resetUpload();

});


// ==========================================
// DRAG AND DROP
// ==========================================

uploadArea.addEventListener("dragover", (event) => {

    event.preventDefault();

    uploadArea.classList.add("dragover");

});


uploadArea.addEventListener("dragleave", () => {

    uploadArea.classList.remove("dragover");

});


uploadArea.addEventListener("drop", (event) => {

    event.preventDefault();

    uploadArea.classList.remove("dragover");

    const file = event.dataTransfer.files[0];

    if (file) {

        imageInput.value = "";

        handleImage(file);

    }

});


// ==========================================
// ANALYZE IMAGE
// ==========================================

analyzeBtn.addEventListener("click", async () => {

    if (!selectedFile) {

        showError("Please select an image first.");

        return;
    }

    hideError();

    setLoading(true);

    resultsSection.classList.add("hidden");

    loadingSection.classList.remove("hidden");

    const formData = new FormData();

    formData.append("image", selectedFile);

    try {

        const response = await fetch("/api/predict", {

            method: "POST",

            body: formData

        });

        const data = await response.json();

        if (!response.ok || data.status !== "success") {

            throw new Error(
                data.message || "Prediction failed. Please try again."
            );

        }

        displayResults(data);

    } catch (error) {

        showError(error.message || "Unable to connect to the server.");

    } finally {

        setLoading(false);

        loadingSection.classList.add("hidden");

    }

});


// ==========================================
// DISPLAY RESULTS
// ==========================================

function displayResults(data) {

    document.getElementById("predictionName").textContent =
        data.prediction;

    document.getElementById("predictionDescription").textContent =
        data.description;

    document.getElementById("confidenceValue").textContent =
        `${Number(data.confidence).toFixed(2)}%`;

    document.getElementById("disclaimerText").textContent =
        data.disclaimer;

    // Status badge

    const statusBadge = document.getElementById("statusBadge");

    statusBadge.textContent = data.plant_status;

    statusBadge.classList.toggle(
        "healthy",
        data.plant_status === "Healthy"
    );

    // Confidence progress bar

    const confidenceBar = document.getElementById("confidenceBar");

    const score = Math.max(
        0,
        Math.min(100, Number(data.confidence) || 0)
    );

    confidenceBar.style.width = "0%";

    requestAnimationFrame(() => {

        requestAnimationFrame(() => {

            confidenceBar.style.width = `${score}%`;

        });

    });

    // Management recommendations

    const managementList = document.getElementById("managementList");

    managementList.replaceChildren();

    data.management.forEach((recommendation) => {

        const item = document.createElement("li");

        item.textContent = recommendation;

        managementList.appendChild(item);

    });

    // Show results

    resultsSection.classList.remove("hidden");

    resultsSection.scrollIntoView({
        behavior: "smooth",
        block: "start"
    });

}


// ==========================================
// LOADING STATE
// ==========================================

function setLoading(isLoading) {

    analyzeBtn.disabled = isLoading || !selectedFile;

    buttonText.classList.toggle("hidden", isLoading);

    buttonLoader.classList.toggle("hidden", !isLoading);

}


// ==========================================
// RESET UPLOAD
// ==========================================

function resetUpload() {

    selectedFile = null;

    imageInput.value = "";

    imagePreview.src = "";

    previewContainer.classList.add("hidden");

    uploadPlaceholder.classList.remove("hidden");

    analyzeBtn.disabled = true;

    resultsSection.classList.add("hidden");

    hideError();

}


// ==========================================
// NEW ANALYSIS
// ==========================================

newAnalysisBtn.addEventListener("click", () => {

    resetUpload();

    window.scrollTo({
        top: 0,
        behavior: "smooth"
    });

});


// ==========================================
// ERROR HANDLING
// ==========================================

function showError(message) {

    errorMessage.textContent = message;

    errorMessage.classList.remove("hidden");

}


function hideError() {

    errorMessage.textContent = "";

    errorMessage.classList.add("hidden");

}