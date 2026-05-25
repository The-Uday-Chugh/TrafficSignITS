const viewHome = document.getElementById("viewHome");
const panelDetect = document.getElementById("panelDetect");

const fileInput = document.getElementById("fileInput");
const uploadBtn = document.getElementById("uploadBtn");
const analyzeBtn = document.getElementById("analyzeBtn");
const analyzeLabel = document.getElementById("analyzeLabel");
const fileLabel = document.getElementById("fileLabel");
const status = document.getElementById("status");
const dropArea = document.getElementById("dropArea");
const dropPlaceholder = document.getElementById("dropPlaceholder");
const previewImg = document.getElementById("previewImg");
const resultsEmpty = document.getElementById("resultsEmpty");
const results = document.getElementById("results");
const resultBadge = document.getElementById("resultBadge");
const imgOriginal = document.getElementById("imgOriginal");
const imgAnnotated = document.getElementById("imgAnnotated");
const figAnnotated = document.getElementById("figAnnotated");
const detectionList = document.getElementById("detectionList");
const progressBarContainer = document.getElementById("progressBarContainer");
const progressBar = document.getElementById("progressBar");

let selectedFile = null;

const btnBackHomeHeader = document.getElementById("btnBackHomeHeader");

function showPanel(name) {
  viewHome.hidden = true;
  panelDetect.hidden = name !== "detect";
  if (btnBackHomeHeader) {
    btnBackHomeHeader.style.display = name === "detect" ? "block" : "none";
  }
  if (name === "detect") {
    const brand = document.querySelector(".brand");
    if (brand) {
      brand.style.display = "none";
    }
    const footerP = document.querySelector(".site-footer p");
    if (footerP) {
      footerP.textContent = "© 2026. AI-Powered Road Sign Recognition for Intelligent Transport Systems.";
    }
  }
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function showHome() {
  viewHome.hidden = false;
  panelDetect.hidden = true;
  if (btnBackHomeHeader) {
    btnBackHomeHeader.style.display = "none";
  }
  const brand = document.querySelector(".brand");
  if (brand) {
    brand.style.display = "flex";
  }
  const footerP = document.querySelector(".site-footer p");
  if (footerP) {
    footerP.textContent = "© 2026. AI-Powered Road Sign Recognition for Intelligent Transport Systems.";
  }
  window.scrollTo({ top: 0, behavior: "smooth" });
}

document.querySelectorAll("[data-open]").forEach((btn) => {
  btn.addEventListener("click", () => showPanel(btn.dataset.open));
});

document.querySelectorAll("[data-close-panel]").forEach((btn) => {
  btn.addEventListener("click", showHome);
});

function setFile(file) {
  selectedFile = file;
  fileLabel.textContent = file.name;
  analyzeBtn.disabled = false;
  status.textContent = "";
  status.className = "status";

  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewImg.hidden = false;
  dropPlaceholder.hidden = true;

  results.hidden = true;
  resultsEmpty.hidden = false;
}

uploadBtn.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (file) setFile(file);
});

["dragenter", "dragover"].forEach((evt) => {
  dropArea.addEventListener(evt, (e) => {
    e.preventDefault();
    dropArea.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((evt) => {
  dropArea.addEventListener(evt, (e) => {
    e.preventDefault();
    dropArea.classList.remove("dragover");
  });
});

dropArea.addEventListener("drop", (e) => {
  const file = e.dataTransfer?.files?.[0];
  if (file && file.type.startsWith("image/")) setFile(file);
});

dropArea.addEventListener("click", () => fileInput.click());

analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  const form = new FormData();
  form.append("image", selectedFile);

  analyzeBtn.disabled = true;
  analyzeLabel.textContent = "Analyzing…";
  status.textContent = "Running YOLO detection…";
  status.className = "status";

  // Show progress bar & trigger reflow to start transition
  progressBarContainer.hidden = false;
  progressBar.classList.remove("active");
  progressBar.offsetWidth; // trigger reflow
  setTimeout(() => {
    progressBar.classList.add("active");
  }, 15);

  const startTime = Date.now();

  try {
    const res = await fetch("/api/predict", { method: "POST", body: form });
    const data = await res.json();

    if (!res.ok) {
      status.textContent = data.detail || "Analysis failed.";
      status.className = "status err";
      return;
    }

    // Enforce 1.5s artificial delay for presentation styling
    const elapsed = Date.now() - startTime;
    const remaining = 1500 - elapsed;
    if (remaining > 0) {
      await new Promise(r => setTimeout(r, remaining));
    }

    resultsEmpty.hidden = true;
    results.hidden = false;
    resultBadge.textContent = `${data.count} sign${data.count === 1 ? "" : "s"} detected`;
    imgOriginal.src = data.original_image;

    if (data.annotated_image) {
      imgAnnotated.src = data.annotated_image;
      figAnnotated.hidden = false;
    } else {
      figAnnotated.hidden = true;
    }

    detectionList.innerHTML = "";
    if (!data.detections.length) {
      detectionList.innerHTML =
        '<li><span class="det-name">No signs found</span></li>';
    } else {
      data.detections.forEach((d, i) => {
        const pct = Math.round(d.confidence * 100);
        const li = document.createElement("li");
        li.innerHTML = `
          <div>
            <span class="det-name">${i + 1}. ${escapeHtml(d.class_name)}</span>
            <div class="conf-bar"><span style="width:${pct}%"></span></div>
          </div>
          <span class="det-conf">${pct}%</span>
        `;
        detectionList.appendChild(li);
      });
    }

    status.textContent = "Analysis complete.";
    status.className = "status ok";
  } catch {
    status.textContent = "Cannot reach server. Run: python main.py";
    status.className = "status err";
  } finally {
    progressBarContainer.hidden = true;
    progressBar.classList.remove("active");
    analyzeBtn.disabled = false;
    analyzeLabel.textContent = "Analyze image";
  }
});

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}
