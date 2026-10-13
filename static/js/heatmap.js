console.log("🔥 LungVision heatmap.js loaded");

const data = sessionStorage.getItem("lungVisionHeatmapData");

if (!data) {
    console.error("❌ No heatmap data found");
} else {

    const heatmapData = JSON.parse(data);

    console.log("🔥 Heatmap data:", heatmapData);

    document.getElementById("originalImg").src =
        heatmapData.original;

    document.getElementById("heatmapImg").src =
        heatmapData.heatmap;

    document.getElementById("overlayImg").src =
        heatmapData.overlay;

    const diseaseElement =
        document.getElementById("diseaseValue");

    const confidenceElement =
        document.getElementById("confidenceValue");

    if (diseaseElement) {
        diseaseElement.textContent =
            heatmapData.disease;
    }

    if (confidenceElement) {
        confidenceElement.textContent =
            heatmapData.confidence + "%";
    }

    console.log(
        "✅ Heatmap loaded for:",
        heatmapData.disease
    );
}

// =====================================================
// BACK BUTTON
// =====================================================

if (btnBack) {

    btnBack.addEventListener("click", () => {

        window.location.href = "/predict";

    });

}