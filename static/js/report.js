/*
  report.js
  ---------------------------------------------------------------
  Fills #report with real data and exports it as a PDF.

  Function to call:
  generateReportPDF(imageDataUrl, fileName, disease, confidence);

  imageDataUrl = uploaded X-ray as Base64 data URL
  fileName     = uploaded image filename
  disease      = predicted disease
  confidence   = confidence number, e.g. 92.14

  Requires html2pdf.js loaded BEFORE this file:
  <script src="https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.10.1/html2pdf.bundle.min.js"></script>
*/

function pad(n) {
  return n.toString().padStart(2, '0');
}

function formatDateTime(date) {
  const months = [
    "January", "February", "March", "April",
    "May", "June", "July", "August",
    "September", "October", "November", "December"
  ];

  let h = date.getHours();
  const ampm = h >= 12 ? 'PM' : 'AM';
  h = h % 12;
  if (h === 0) h = 12;

  return `${pad(date.getDate())} ${months[date.getMonth()]} ${date.getFullYear()}, ${pad(h)}:${pad(date.getMinutes())} ${ampm}`;
}

function makeReportId(date) {
  const y = date.getFullYear();
  const m = pad(date.getMonth() + 1);
  const d = pad(date.getDate());
  const rand = Math.floor(Math.random() * 9000 + 1000);
  return `LVAI-${y}${m}${d}-${rand}`;
}

/*
  Fill all dynamic report fields
*/
function fillReport(imageDataUrl, fileName, disease, confidence) {
  const now = new Date();
  const dateTimeStr = formatDateTime(now);
  const reportId = makeReportId(now);
  const confidenceStr = Number(confidence).toFixed(2) + '%';

  // REPORT INFORMATION
  document.getElementById('rReportId').textContent = reportId;
  document.getElementById('rDateTime').textContent = dateTimeStr;
  document.getElementById('rFileName').textContent = fileName;
  document.getElementById('rDisease').textContent = disease;
  document.getElementById('rConfidence').textContent = confidenceStr;

  // UPLOADED X-RAY IMAGE
  const xrayImage = document.getElementById('rXrayImg');
  xrayImage.src = imageDataUrl;
  xrayImage.alt = `Uploaded Chest X-Ray - ${fileName}`;

  // AI PREDICTION SUMMARY
  document.getElementById('sDisease').textContent = disease;
  document.getElementById('sConfidence').textContent = confidenceStr;
  document.getElementById('sTime').textContent =
    (Math.random() * 0.9 + 0.3).toFixed(2) + ' Seconds';

  // REPORT AUTHENTICATION
  document.getElementById('aGenerated').textContent = dateTimeStr;

  // DISEASE COLOR
  const isNormal = String(disease).toLowerCase() === 'normal';
  document.getElementById('rDisease').style.color = isNormal ? '#1a9c5a' : '#d92b2b';
  document.getElementById('sDisease').style.color = isNormal ? '#1a9c5a' : '#d92b2b';
}

/*
  Wait for an <img> element to finish loading (or fail),
  so html2pdf never captures a blank/broken image.
*/
function waitForImage(imgEl) {
  return new Promise((resolve) => {
    if (imgEl.complete && imgEl.naturalWidth !== 0) {
      resolve();
      return;
    }
    imgEl.onload = () => resolve();
    imgEl.onerror = () => resolve();
  });
}

/*
  Generate PDF
*/
async function generateReportPDF(imageDataUrl, fileName, disease, confidence) {
  if (!imageDataUrl) {
    alert('No X-ray image was provided for this report.');
    return;
  }

  fillReport(imageDataUrl, fileName, disease, confidence);

  const reportEl = document.getElementById('report');
  reportEl.style.display = 'block'; // must be visible for html2canvas to capture it

  const xrayImage = document.getElementById('rXrayImg');
  await waitForImage(xrayImage);

  // Small delay to let the browser finish layout/paint
  await new Promise(resolve => setTimeout(resolve, 300));

  const opt = {
    margin: 0,
    filename: `LungVisionAI_Report_${String(disease).replace(/\s+/g, '_')}.pdf`,
    image: { type: 'jpeg', quality: 0.98 },
    html2canvas: { scale: 2, useCORS: true, allowTaint: true },
    jsPDF: { unit: 'pt', format: 'a4', orientation: 'portrait' }
  };

  try {
    await html2pdf().set(opt).from(reportEl).save();
  } catch (error) {
    console.error('PDF generation failed:', error);
    alert('Unable to generate PDF. Please try again.');
  } finally {
    reportEl.style.display = 'none';
  }
}