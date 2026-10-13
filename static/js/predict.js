// =====================================================
// LUNGVISION AI - PREDICTION JAVASCRIPT
// =====================================================
 
console.log("LungVision AI Predict Loaded");
 
 
// =====================================================
// SHARED STATE FOR PDF REPORT
// =====================================================
 
let uploadedImageDataUrl = null;
let uploadedFileName = null;
 
let predictedDisease = null;
let predictedConfidence = null;
 
let lastPredictionData = null;
 
 
// =====================================================
// ELEMENTS
// =====================================================
 
const fileInput =
    document.getElementById("fileInput");
 
const dropArea =
    document.getElementById("dropArea");
 
const previewImage =
    document.getElementById("previewImage");
 
const previewPlaceholder =
    document.getElementById("previewPlaceholder");
 
const analyzeBtn =
    document.getElementById("analyzeBtn");
 
const resetBtn =
    document.getElementById("resetBtn");
 
const loader =
    document.getElementById("loader");
 
const resultCard =
    document.getElementById("resultCard");
 
const prediction =
    document.getElementById("prediction");
 
const accuracy =
    document.getElementById("accuracy");
 
const status =
    document.getElementById("status");
 
const timeTaken =
    document.getElementById("timeTaken");
 
const generateReportBtn =
    document.getElementById("generateReportBtn");
 
const generateCouponBtn =
    document.getElementById("generateCouponBtn");
 
// FIX: this element was referenced throughout the file but never
// actually grabbed from the DOM, causing a ReferenceError that
// silently broke the analyze click handler.
const generateHeatmapBtn =
    document.getElementById("btnHeatmap");
 
 
// =====================================================
// PRECAUTIONS CARD ELEMENTS (NEW)
// =====================================================
 
const precautionsCard =
    document.getElementById("precautionsCard");
 
const precautionsList =
    document.getElementById("precautionsList");
 
const speakBtn =
    document.getElementById("speakBtn");
 
const langTabs =
    document.querySelectorAll(".lang-tab");
 
let currentPrecautionsData = null;   // { en: [...], hi: [...], mr: [...] }
let currentLang = "en";
 
 
// =====================================================
// IMAGE PREVIEW
// =====================================================
 
fileInput.addEventListener(
    "change",
    previewFile
);
 
 
function previewFile() {
 
    const file =
        fileInput.files[0];
 
    if (!file) {
        return;
    }
 
 
    // Store file name
 
    uploadedFileName =
        file.name;
 
 
    // Read image
 
    const reader =
        new FileReader();
 
 
    reader.onload =
        function (event) {
 
            uploadedImageDataUrl =
                event.target.result;
 
            previewImage.src =
                event.target.result;
 
            previewImage.style.display =
                "block";
 
 
            if (previewPlaceholder) {
 
                previewPlaceholder.style.display =
                    "none";
 
            }
 
        };
 
 
    reader.readAsDataURL(file);
 
}
 
 
// =====================================================
// DRAG & DROP
// =====================================================
 
[
    "dragenter",
    "dragover"
].forEach(
    eventName => {
 
        dropArea.addEventListener(
            eventName,
            event => {
 
                event.preventDefault();
 
                dropArea.classList.add(
                    "dragover"
                );
 
            }
        );
 
    }
);
 
 
[
    "dragleave",
    "drop"
].forEach(
    eventName => {
 
        dropArea.addEventListener(
            eventName,
            event => {
 
                event.preventDefault();
 
                dropArea.classList.remove(
                    "dragover"
                );
 
            }
        );
 
    }
);
 
 
dropArea.addEventListener(
    "drop",
    event => {
 
        const files =
            event.dataTransfer.files;
 
 
        if (files.length > 0) {
 
            fileInput.files =
                files;
 
            previewFile();
 
        }
 
    }
);
 
 
// =====================================================
// ANALYZE IMAGE
// =====================================================
 
analyzeBtn.addEventListener(
    "click",
    async () => {
 
 
        // -------------------------------------------------
        // Check image
        // -------------------------------------------------
 
        if (
            fileInput.files.length === 0
        ) {
 
            alert(
                "Please select a chest X-ray image."
            );
 
            return;
 
        }
 
 
       // -------------------------------------------------
// Reset previous results
// -------------------------------------------------
 
loader.classList.add("active");
 
resultCard.classList.remove("show");
 
generateReportBtn.style.display = "none";
 
if (generateHeatmapBtn) {
    generateHeatmapBtn.style.display = "flex";
}
 
if (generateCouponBtn) {
    generateCouponBtn.style.display = "none";
}
 
if (precautionsCard) {
    precautionsCard.style.display = "none";
}
 
 
        // -------------------------------------------------
        // Start timer
        // -------------------------------------------------
 
        const start =
            performance.now();
 
 
        // -------------------------------------------------
        // Form data
        // -------------------------------------------------
 
        const formData =
            new FormData();
 
        formData.append(
            "image",
            fileInput.files[0]
        );
 
 
        try {
 
 
            // -------------------------------------------------
            // Send image to Flask
            // -------------------------------------------------
 
            const response =
                await fetch(
                    "/api/predict",
                    {
 
                        method: "POST",
 
                        credentials:
                            "same-origin",
 
                        body:
                            formData
 
                    }
                );
 
 
            // -------------------------------------------------
            // Server error
            // -------------------------------------------------
 
            if (!response.ok) {
 
                throw new Error(
                    "Server Error"
                );
 
            }
 
 
            const data =
                await response.json();
 
 
            // -------------------------------------------------
            // Stop loader
            // -------------------------------------------------
 
            loader.classList.remove(
                "active"
            );
 
            resultCard.classList.add(
                "show"
            );
 
 
            // =================================================
            // BACKEND REJECTION
            // =================================================
 
            if (data.error) {
 
 
                prediction.innerText =
                    "Rejected";
 
 
                accuracy.innerText =
                    "--";
 
 
                status.innerHTML =
                    '<span class="error">' +
                    data.error +
                    '</span>';
 
 
                const end =
                    performance.now();
 
 
                timeTaken.innerText =
                    (
                        (end - start) /
                        1000
                    ).toFixed(2) +
                    " sec";
 
 
                generateReportBtn.style.display =
                    "none";
 
 
                if (generateCouponBtn) {
 
                    generateCouponBtn.style.display =
                        "none";
 
                }
 
                if (generateHeatmapBtn) {
 
                    generateHeatmapBtn.style.display =
                        "none";
 
                }
 
                if (precautionsCard) {
 
                    precautionsCard.style.display =
                        "none";
 
                }
 
 
                return;
 
            }
 
 
            // =================================================
            // SUCCESSFUL PREDICTION
            // =================================================
 
 
            // Disease
 
            predictedDisease =
                data.prediction ||
                "Unknown";
 
 
            // Confidence
 
            predictedConfidence =
                parseFloat(
                    data.confidence ?? 0
                );
 
 
            prediction.innerText =
                predictedDisease;
 
 
            accuracy.innerText =
                predictedConfidence +
                "%";
 
 
            // Status
 
            status.innerHTML =
                '<span class="success">' +
                'Analysis Complete' +
                '</span>';
 
 
            // -------------------------------------------------
            // Analysis time
            // -------------------------------------------------
 
            const end =
                performance.now();
 
 
            const analysisTime =
                (
                    (end - start) /
                    1000
                ).toFixed(2) +
                " sec";
 
 
            timeTaken.innerText =
                analysisTime;
 
 
            // =================================================
            // SAVE PREDICTION DATA
            // =================================================
 
            lastPredictionData = {
 
                disease:
                    predictedDisease,
 
                confidence:
                    predictedConfidence,
 
                analysis_time:
                    analysisTime
 
            };
 
 
            // =================================================
            // SHOW PDF BUTTON
            // =================================================
 
            generateReportBtn.style.display =
                "flex";
 
 
            // =================================================
            // LOAD HOME CARE PRECAUTIONS (NEW)
            // =================================================
 
            loadPrecautions(predictedDisease);
 
 
            // =================================================
            // COUPON LOGIC
            // =================================================
 
            if (
    predictedConfidence > 85
) {

    console.log(
        "Coupon eligible:",
        predictedConfidence
    );


    generateCouponBtn.style.display =
        "flex";


    // Store prediction so coupon page can use it

    sessionStorage.setItem(
        "lungVisionCouponData",
        JSON.stringify({
            disease: predictedDisease,
            confidence: predictedConfidence
        })
    );


} else {


    console.log(
        "Coupon not eligible:",
        predictedConfidence
    );


    generateCouponBtn.style.display =
        "none";


    sessionStorage.removeItem(
        "lungVisionCouponData"
    );

}
 
 
        }
 
        catch (error) {
 
 
            console.error(error);
 
 
            loader.classList.remove(
                "active"
            );
 
 
            resultCard.classList.add(
                "show"
            );
 
 
            prediction.innerText =
                "--";
 
 
            accuracy.innerText =
                "--";
 
 
            timeTaken.innerText =
                "--";
 
 
            status.innerHTML =
                '<span class="error">' +
                'Connection Failed' +
                '</span>';
 
 
            generateReportBtn.style.display =
                "none";
 
 
            if (generateCouponBtn) {
 
                generateCouponBtn.style.display =
                    "none";
 
            }
 
 
            alert(
                "Cannot connect to Flask backend."
            );
 
        }
 
    }
);
 
 
// =====================================================
// GENERATE PDF REPORT
// =====================================================
 
generateReportBtn.addEventListener(
    "click",
    async () => {
 
 
        if (!lastPredictionData) {
 
            return;
 
        }
 
 
        generateReportBtn.disabled =
            true;
 
 
        generateReportBtn.innerHTML =
            '<i class="fa-solid fa-spinner fa-spin"></i> Generating...';
 
 
        try {
 
 
            const response =
                await fetch(
                    "/api/generate-report",
                    {
 
                        method: "POST",
 
                        credentials:
                            "same-origin",
 
                        headers: {
 
                            "Content-Type":
                                "application/json"
 
                        },
 
                        body:
                            JSON.stringify(
                                lastPredictionData
                            )
 
                    }
                );
 
 
            if (!response.ok) {
 
                throw new Error(
                    "Failed to generate report"
                );
 
            }
 
 
            const blob =
                await response.blob();
 
 
            const url =
                window.URL.createObjectURL(
                    blob
                );
 
 
            const a =
                document.createElement(
                    "a"
                );
 
 
            a.href =
                url;
 
 
            a.download =
                "LungVision_Report.pdf";
 
 
            document.body.appendChild(a);
 
 
            a.click();
 
 
            a.remove();
 
 
            window.URL.revokeObjectURL(
                url
            );
 
 
        }
 
        catch (error) {
 
 
            console.error(error);
 
 
            alert(
                "Could not generate the PDF report."
            );
 
        }
 
        finally {
 
 
            generateReportBtn.disabled =
                false;
 
 
            generateReportBtn.innerHTML =
                '<i class="fa-solid fa-file-pdf"></i> Generate PDF Report';
 
        }
 
    }
);
 
 
// =====================================================
// RESET
// =====================================================
 
resetBtn.addEventListener(
    "click",
    () => {
 
 
        // Clear file
 
        fileInput.value =
            "";
 
 
        // Clear preview
 
        previewImage.src =
            "";
 
        previewImage.style.display =
            "none";
 
 
        if (previewPlaceholder) {
 
            previewPlaceholder.style.display =
                "block";
 
        }
 
 
        // Clear prediction
 
        prediction.innerText =
            "--";
 
 
        accuracy.innerText =
            "--";
 
 
        status.innerText =
            "Waiting";
 
 
        timeTaken.innerText =
            "--";
 
 
        // Hide cards
 
        resultCard.classList.remove(
            "show"
        );
 
 
        loader.classList.remove(
            "active"
        );
 
 
        // Hide buttons
 
        generateReportBtn.style.display =
            "none";
 
 
        if (generateCouponBtn) {
 
            generateCouponBtn.style.display =
                "none";
 
        }
 
        if (generateHeatmapBtn) {
 
            generateHeatmapBtn.style.display =
                "none";
 
        }
 
        // Hide precautions card (NEW)
 
        if (precautionsCard) {
 
            precautionsCard.style.display =
                "none";
 
        }
 
        currentPrecautionsData =
            null;


            // Stop any playing audio on reset

if (currentAudio) {

    currentAudio.pause();
    currentAudio.currentTime = 0;

    isSpeaking = false;

}
 
 
        // Clear state
 
        lastPredictionData =
            null;
 
        uploadedImageDataUrl =
            null;
 
        uploadedFileName =
            null;
 
        predictedDisease =
            null;
 
        predictedConfidence =
            null;
 
 
        // Clear coupon session data
 
        sessionStorage.removeItem(
            "lungVisionCouponData"
        );
 
        // Clear heatmap session data
 
        sessionStorage.removeItem(
            "lungVisionHeatmapData"
        );
 
    }
);
 
 
// =====================================================
// REAL GRAD-CAM HEATMAP
// =====================================================
 
if (generateHeatmapBtn) {
 
    generateHeatmapBtn.addEventListener(
        "click",
        async function () {
 
            console.log(
                "🔥 Generate REAL Heatmap clicked!"
            );
 
            // ---------------------------------------------
            // Make sure prediction exists
            // ---------------------------------------------
 
            if (!predictedDisease) {
 
                alert(
                    "Please analyze the X-ray first."
                );
 
                return;
            }
 
            generateHeatmapBtn.disabled = true;
 
            generateHeatmapBtn.innerHTML =
                '<i class="fa-solid fa-spinner fa-spin"></i> Generating Heatmap...';
 
            try {
 
                console.log(
                    "🔥 Using existing prediction:"
                );
 
                console.log(
                    "Disease:",
                    predictedDisease
                );
 
                console.log(
                    "Confidence:",
                    predictedConfidence
                );
 
                // ---------------------------------------------
                // IMPORTANT:
                // DO NOT upload the image again.
                //
                // Flask already has the analyzed X-ray
                // and the exact prediction in the session.
                // ---------------------------------------------
 
                const response =
                    await fetch(
                        "/generate-heatmap",
                        {
                            method: "POST",
                            credentials: "same-origin"
                        }
                    );
 
                console.log(
                    "📥 Heatmap response status:",
                    response.status
                );
 
                const data =
                    await response.json();
 
                console.log(
                    "🔥 Grad-CAM response:",
                    data
                );
 
                // ---------------------------------------------
                // Error
                // ---------------------------------------------
 
                if (
                    !response.ok ||
                    data.error
                ) {
 
                    throw new Error(
                        data.error ||
                        "Heatmap generation failed."
                    );
 
                }
 
                // ---------------------------------------------
                // Make sure Flask returned everything
                // ---------------------------------------------
 
                if (
                    !data.original ||
                    !data.heatmap ||
                    !data.overlay
                ) {
 
                    throw new Error(
                        "Heatmap images were not returned by Flask."
                    );
 
                }
 
                // ---------------------------------------------
                // Store REAL heatmap data
                // ---------------------------------------------
 
                const heatmapData = {
 
                    disease:
                        data.disease ||
                        predictedDisease,
 
                    confidence:
                        data.confidence ??
                        predictedConfidence,
 
                    original:
                        data.original,
 
                    heatmap:
                        data.heatmap,
 
                    overlay:
                        data.overlay
 
                };
 
                sessionStorage.setItem(
                    "lungVisionHeatmapData",
                    JSON.stringify(heatmapData)
                );
 
                console.log(
                    "✅ Heatmap data saved:"
                );
 
                console.log(
                    heatmapData
                );
 
                // ---------------------------------------------
                // Open heatmap ONLY after generation succeeds
                // ---------------------------------------------
 
                window.location.href =
                    "/heatmap";
 
            }
 
            catch (error) {
 
                console.error(
                    "❌ Heatmap Error:",
                    error
                );
 
                alert(
                    "Heatmap generation failed:\n\n" +
                    error.message
                );
 
            }
 
            finally {
 
                generateHeatmapBtn.disabled =
                    false;
 
                generateHeatmapBtn.innerHTML =
                    '<i class="fa-solid fa-fire"></i> Generate Heatmap';
 
            }
 
        }
    );
 
}
 
 
// =====================================================
// PRECAUTIONS + LANGUAGE TABS + TEXT-TO-SPEECH (NEW)
// =====================================================
 
async function loadPrecautions(disease) {
 
    if (!precautionsCard || !precautionsList) {
        return;
    }
 
    try {
 
        const response =
            await fetch(`/api/precautions/${disease}`);
 
        if (!response.ok) {
 
            precautionsCard.style.display =
                "none";
 
            return;
 
        }
 
        const data =
            await response.json();
 
        currentPrecautionsData =
            data.precautions;
 
        currentLang =
            "en";
 
        // Reset tabs to English active
 
        langTabs.forEach(
            tab => tab.classList.remove("active")
        );
 
        const englishTab =
            document.querySelector('.lang-tab[data-lang="en"]');
 
        if (englishTab) {
 
            englishTab.classList.add("active");
 
        }
 
        renderPrecautions(currentLang);
 
        precautionsCard.style.display =
            "block";
 
    }
 
    catch (error) {
 
        console.error(
            "Precautions fetch error:",
            error
        );
 
        precautionsCard.style.display =
            "none";
 
    }
 
}
 
 
function renderPrecautions(lang) {
 
    if (!precautionsList) {
        return;
    }
 
    if (
        !currentPrecautionsData ||
        !currentPrecautionsData[lang]
    ) {
 
        precautionsList.innerHTML =
            "";
 
        return;
 
    }
 
    precautionsList.innerHTML =
        "";
 
    currentPrecautionsData[lang].forEach(
        item => {
 
            const li =
                document.createElement("li");
 
            li.innerHTML =
                `<i class="fa-solid fa-check"></i> ${item}`;
 
            precautionsList.appendChild(li);
 
        }
    );
 
}
 
 
// Language tab click handling
 
langTabs.forEach(
    tab => {
 
       tab.addEventListener(
    "click",
    () => {

        langTabs.forEach(
            t => t.classList.remove("active")
        );

        tab.classList.add(
            "active"
        );

        currentLang =
            tab.getAttribute("data-lang");

        // Stop any playing audio when language changes

        if (currentAudio) {

            currentAudio.pause();
            currentAudio.currentTime = 0;

            isSpeaking = false;

            speakBtn.innerHTML =
                '<i class="fa-solid fa-volume-high"></i> Listen';

        }

        renderPrecautions(currentLang);

    }
);
 
    }
);
 
 
// Speak button click handling
 
// =====================================================
// SPEAK / STOP TOGGLE
// =====================================================

let currentAudio = null;
let isSpeaking = false;

if (speakBtn) {

    speakBtn.addEventListener(
        "click",
        async () => {

            // -------------------------------------------------
            // If already speaking, STOP it
            // -------------------------------------------------

            if (isSpeaking && currentAudio) {

                currentAudio.pause();
                currentAudio.currentTime = 0;

                isSpeaking = false;

                speakBtn.innerHTML =
                    '<i class="fa-solid fa-volume-high"></i> Listen';

                return;

            }

            // -------------------------------------------------
            // Otherwise, START speaking
            // -------------------------------------------------

            if (
                !currentPrecautionsData ||
                !currentPrecautionsData[currentLang]
            ) {

                return;

            }

            const textToSpeak =
                currentPrecautionsData[currentLang].join(". ");

            speakBtn.disabled =
                true;

            speakBtn.innerHTML =
                '<i class="fa-solid fa-spinner fa-spin"></i> Loading...';

            try {

                const response =
                    await fetch(
                        "/api/text-to-speech",
                        {

                            method: "POST",

                            headers: {
                                "Content-Type": "application/json"
                            },

                            body: JSON.stringify({
                                text: textToSpeak,
                                lang: currentLang
                            })

                        }
                    );

                if (!response.ok) {

                    throw new Error(
                        "TTS failed"
                    );

                }

                const audioBlob =
                    await response.blob();

                const audioUrl =
                    URL.createObjectURL(audioBlob);

                currentAudio =
                    new Audio(audioUrl);

                // When audio finishes naturally, reset button

                currentAudio.addEventListener(
                    "ended",
                    () => {

                        isSpeaking = false;

                        speakBtn.innerHTML =
                            '<i class="fa-solid fa-volume-high"></i> Listen';

                    }
                );

                currentAudio.play();

                isSpeaking = true;

                speakBtn.disabled =
                    false;

                speakBtn.innerHTML =
                    '<i class="fa-solid fa-stop"></i> Stop';

            }

            catch (error) {

                console.error(
                    "Speech error:",
                    error
                );

                alert(
                    "Could not generate speech."
                );

                isSpeaking = false;

                speakBtn.innerHTML =
                    '<i class="fa-solid fa-volume-high"></i> Listen';

            }

            finally {

                speakBtn.disabled =
                    false;

            }

        }
    );

}