const THEME_KEY = "face-signal-lab-theme";
const themeToggle = document.getElementById("theme-toggle");

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  themeToggle.textContent = theme === "dark" ? "☀" : "☾";
  themeToggle.setAttribute("aria-label", theme === "dark" ? "Açık temaya geç" : "Koyu temaya geç");
}

function getStoredTheme() {
  try {
    return localStorage.getItem(THEME_KEY);
  } catch (err) {
    return null;
  }
}

function storeTheme(theme) {
  try {
    localStorage.setItem(THEME_KEY, theme);
  } catch (err) {
    // Private browsing or storage disabled — theme just won't persist.
  }
}

applyTheme(getStoredTheme() || "dark");

themeToggle.addEventListener("click", () => {
  const next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
  applyTheme(next);
  storeTheme(next);
});

const tabButtons = document.querySelectorAll(".tab-button");
const tabPanels = {
  "tab-detect": document.getElementById("panel-detect"),
  "tab-swap": document.getElementById("panel-swap"),
};

tabButtons.forEach((button) => {
  button.addEventListener("click", () => {
    tabButtons.forEach((candidate) => {
      const isActive = candidate === button;
      candidate.classList.toggle("active", isActive);
      candidate.setAttribute("aria-selected", String(isActive));
    });
    Object.entries(tabPanels).forEach(([id, panel]) => {
      panel.hidden = id !== button.id;
    });
  });
});

function setReadout(dl, pairs) {
  dl.innerHTML = "";
  for (const [label, value] of pairs) {
    const pair = document.createElement("span");
    pair.className = "pair";
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    dd.textContent = value;
    pair.append(dt, dd);
    dl.append(pair);
  }
  dl.hidden = pairs.length === 0;
}

function showImage(imgEl, frameEl, src) {
  imgEl.src = src;
  imgEl.hidden = false;
  frameEl.classList.add("has-image");
}

const form = document.getElementById("upload-form");
const statusEl = document.getElementById("status");
const readoutEl = document.getElementById("readout");
const originalFrame = document.getElementById("original-frame");
const originalImage = document.getElementById("original-image");
const resultFrame = document.getElementById("result-frame");
const resultImage = document.getElementById("result-image");
let originalImageObjectUrl = null;

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const fileInput = document.getElementById("image-input");
  if (!fileInput.files.length) {
    return;
  }
  const file = fileInput.files[0];

  if (originalImageObjectUrl) {
    URL.revokeObjectURL(originalImageObjectUrl);
  }
  originalImageObjectUrl = URL.createObjectURL(file);
  showImage(originalImage, originalFrame, originalImageObjectUrl);

  const formData = new FormData();
  formData.append("image", file);

  statusEl.textContent = "İşleniyor...";
  setReadout(readoutEl, []);
  resultImage.hidden = true;
  resultFrame.classList.remove("has-image");

  let response;
  let body;
  try {
    response = await fetch("/detect", { method: "POST", body: formData });
    body = await response.json();
  } catch (err) {
    statusEl.textContent = "Hata: sunucudan beklenmeyen bir yanıt geldi.";
    return;
  }

  if (!response.ok) {
    statusEl.textContent = `Hata: ${body.error}`;
    return;
  }

  statusEl.textContent = "Yüz ve landmark tespit edildi.";
  setReadout(readoutEl, [
    ["LANDMARK", String(body.landmark_count)],
    ["YAW", `${body.head_pose.yaw.toFixed(1)}°`],
    ["PITCH", `${body.head_pose.pitch.toFixed(1)}°`],
    ["ROLL", `${body.head_pose.roll.toFixed(1)}°`],
    ["SÜRE", `${body.processing_time_seconds.toFixed(2)}s`],
  ]);
  showImage(resultImage, resultFrame, `data:image/png;base64,${body.annotated_image_png_base64}`);
});

const swapForm = document.getElementById("swap-form");
const swapStatusEl = document.getElementById("swap-status");
const swapReadoutEl = document.getElementById("swap-readout");
const swapOriginalFrame = document.getElementById("swap-original-frame");
const swapOriginalImage = document.getElementById("swap-original-image");
const swapResultFrame = document.getElementById("swap-result-frame");
const swapResultImage = document.getElementById("swap-result-image");
let swapOriginalImageObjectUrl = null;

swapForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const sourceInput = document.getElementById("source-input");
  const targetInput = document.getElementById("target-input");
  if (!sourceInput.files.length || !targetInput.files.length) {
    return;
  }
  const targetFile = targetInput.files[0];

  if (swapOriginalImageObjectUrl) {
    URL.revokeObjectURL(swapOriginalImageObjectUrl);
  }
  swapOriginalImageObjectUrl = URL.createObjectURL(targetFile);
  showImage(swapOriginalImage, swapOriginalFrame, swapOriginalImageObjectUrl);

  const formData = new FormData();
  formData.append("source", sourceInput.files[0]);
  formData.append("target", targetFile);

  swapStatusEl.textContent = "İşleniyor...";
  setReadout(swapReadoutEl, []);
  swapResultImage.hidden = true;
  swapResultFrame.classList.remove("has-image");

  let response;
  let body;
  try {
    response = await fetch("/swap", { method: "POST", body: formData });
    body = await response.json();
  } catch (err) {
    swapStatusEl.textContent = "Hata: sunucudan beklenmeyen bir yanıt geldi.";
    return;
  }

  if (!response.ok) {
    swapStatusEl.textContent = `Hata: ${body.error}`;
    return;
  }

  const similarityText = body.identity_similarity === null
    ? "n/a"
    : `${(body.identity_similarity * 100).toFixed(1)}%`;
  swapStatusEl.textContent = "Yüz değiştirme tamamlandı.";
  setReadout(swapReadoutEl, [
    ["BENZERLİK", similarityText],
    ["SÜRE", `${body.processing_time_seconds.toFixed(2)}s`],
  ]);
  showImage(swapResultImage, swapResultFrame, `data:image/png;base64,${body.swapped_image_png_base64}`);
});

const swapInputsButton = document.getElementById("swap-inputs-button");

swapInputsButton.addEventListener("click", () => {
  const sourceInput = document.getElementById("source-input");
  const targetInput = document.getElementById("target-input");

  const sourceFiles = sourceInput.files;
  const targetFiles = targetInput.files;

  const newSourceFiles = new DataTransfer();
  Array.from(targetFiles).forEach((file) => newSourceFiles.items.add(file));
  sourceInput.files = newSourceFiles.files;

  const newTargetFiles = new DataTransfer();
  Array.from(sourceFiles).forEach((file) => newTargetFiles.items.add(file));
  targetInput.files = newTargetFiles.files;
});
