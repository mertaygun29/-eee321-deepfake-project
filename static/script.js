const THEME_KEY = "face-signal-lab-theme";
const themeToggle = document.getElementById("theme-toggle");

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  themeToggle.textContent = theme === "dark" ? "☀" : "☾";
  themeToggle.setAttribute("aria-label", theme === "dark" ? "Switch to light theme" : "Switch to dark theme");
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

const tabButtons = document.querySelectorAll(".nav-item");
const tabPanels = {
  "tab-detect": document.getElementById("panel-detect"),
  "tab-swap": document.getElementById("panel-swap"),
  "tab-lipsync": document.getElementById("panel-lipsync"),
  "tab-models": document.getElementById("panel-models"),
};

const TAB_FADE_MS = 160;

function switchTab(targetId) {
  const newPanel = tabPanels[targetId];
  const current = Object.values(tabPanels).find((panel) => !panel.hidden);

  function reveal() {
    newPanel.hidden = false;
    newPanel.classList.add("fade-out");
    void newPanel.offsetWidth; // force reflow so the next class change transitions
    newPanel.classList.remove("fade-out");
  }

  if (current && current !== newPanel) {
    current.classList.add("fade-out");
    window.setTimeout(() => {
      current.hidden = true;
      current.classList.remove("fade-out");
      reveal();
    }, TAB_FADE_MS);
  } else {
    reveal();
  }
}

tabButtons.forEach((button) => {
  button.addEventListener("click", () => {
    tabButtons.forEach((candidate) => {
      const isActive = candidate === button;
      candidate.classList.toggle("active", isActive);
      candidate.setAttribute("aria-selected", String(isActive));
    });
    switchTab(button.id);
  });
});

const systemStatusList = document.getElementById("system-status-list");

function renderStatusRow(dl, label, ok) {
  const row = document.createElement("div");
  row.className = "status-row";
  const dt = document.createElement("dt");
  dt.textContent = label;
  const dd = document.createElement("dd");
  dd.textContent = ok ? "ready" : "not set up";
  dd.className = ok ? "status-ok" : "status-missing";
  row.append(dt, dd);
  dl.append(row);
}

async function loadSystemStatus() {
  let body;
  try {
    const response = await fetch("/status");
    body = await response.json();
  } catch (err) {
    systemStatusList.textContent = "";
    renderStatusRow(systemStatusList, "STATUS", false);
    return;
  }
  systemStatusList.innerHTML = "";
  renderStatusRow(systemStatusList, "MEDIAPIPE", body.mediapipe);
  renderStatusRow(systemStatusList, "INSIGHTFACE", body.insightface);
  renderStatusRow(systemStatusList, "WAV2LIP", body.wav2lip);
  renderStatusRow(systemStatusList, "FFMPEG", body.ffmpeg);
}

loadSystemStatus();

const activityLog = document.getElementById("activity-log");
const MAX_LOG_ENTRIES = 20;

function logActivity(label, detail, isError) {
  const entry = document.createElement("li");
  entry.className = "log-entry" + (isError ? " log-error" : "");

  const timeEl = document.createElement("span");
  timeEl.className = "log-time";
  timeEl.textContent = new Date().toLocaleTimeString([], { hour12: false });

  const labelEl = document.createElement("span");
  labelEl.className = "log-label";
  labelEl.textContent = label;

  const detailEl = document.createElement("span");
  detailEl.className = "log-detail";
  detailEl.textContent = detail;

  entry.append(timeEl, labelEl, detailEl);
  activityLog.prepend(entry);

  while (activityLog.children.length > MAX_LOG_ENTRIES) {
    activityLog.removeChild(activityLog.lastChild);
  }
}

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

const ZOOM_FACTOR = 2.5;

function getContainedImageRect(img, containerWidth, containerHeight) {
  const naturalRatio = img.naturalWidth / img.naturalHeight;
  const containerRatio = containerWidth / containerHeight;
  let width;
  let height;

  if (naturalRatio > containerRatio) {
    width = containerWidth;
    height = containerWidth / naturalRatio;
  } else {
    height = containerHeight;
    width = containerHeight * naturalRatio;
  }

  return {
    width,
    height,
    offsetX: (containerWidth - width) / 2,
    offsetY: (containerHeight - height) / 2,
  };
}

function attachMagnifier(frameEl, imgEl) {
  const lens = document.createElement("div");
  lens.className = "zoom-lens";
  frameEl.appendChild(lens);
  const lensSize = 130;

  function hideLens() {
    lens.style.display = "none";
    frameEl.style.cursor = "";
  }

  frameEl.addEventListener("mousemove", (event) => {
    if (imgEl.hidden || !imgEl.naturalWidth) {
      hideLens();
      return;
    }

    const rect = frameEl.getBoundingClientRect();
    const image = getContainedImageRect(imgEl, rect.width, rect.height);
    const x = event.clientX - rect.left - image.offsetX;
    const y = event.clientY - rect.top - image.offsetY;

    if (x < 0 || y < 0 || x > image.width || y > image.height) {
      hideLens();
      return;
    }

    frameEl.style.cursor = "none";
    lens.style.display = "block";

    let lensX = image.offsetX + x - lensSize / 2;
    let lensY = image.offsetY + y - lensSize / 2;
    lensX = Math.max(0, Math.min(lensX, rect.width - lensSize));
    lensY = Math.max(0, Math.min(lensY, rect.height - lensSize));
    lens.style.left = `${lensX}px`;
    lens.style.top = `${lensY}px`;

    lens.style.backgroundImage = `url(${imgEl.src})`;
    lens.style.backgroundSize = `${image.width * ZOOM_FACTOR}px ${image.height * ZOOM_FACTOR}px`;
    lens.style.backgroundPosition =
      `${-(x * ZOOM_FACTOR - lensSize / 2)}px ${-(y * ZOOM_FACTOR - lensSize / 2)}px`;
  });

  frameEl.addEventListener("mouseleave", hideLens);
}

const form = document.getElementById("upload-form");
const statusEl = document.getElementById("status");
const progressDetect = document.getElementById("progress-detect");
const readoutEl = document.getElementById("readout");
const originalFrame = document.getElementById("original-frame");
const originalImage = document.getElementById("original-image");
const resultFrame = document.getElementById("result-frame");
const resultImage = document.getElementById("result-image");
let originalImageObjectUrl = null;

attachMagnifier(originalFrame, originalImage);
attachMagnifier(resultFrame, resultImage);

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

  statusEl.textContent = "Processing...";
  progressDetect.hidden = false;
  setReadout(readoutEl, []);
  resultImage.hidden = true;
  resultFrame.classList.remove("has-image");

  let response;
  let body;
  try {
    response = await fetch("/detect", { method: "POST", body: formData });
    body = await response.json();
  } catch (err) {
    statusEl.textContent = "Error: unexpected response from server.";
    logActivity("01 DETECT", "Could not read server response", true);
    progressDetect.hidden = true;
    return;
  }

  progressDetect.hidden = true;

  if (!response.ok) {
    statusEl.textContent = `Error: ${body.error}`;
    logActivity("01 DETECT", `Error: ${body.error}`, true);
    return;
  }

  statusEl.textContent = "Face and landmarks detected.";
  setReadout(readoutEl, [
    ["LANDMARKS", String(body.landmark_count)],
    ["YAW", `${body.head_pose.yaw.toFixed(1)}°`],
    ["PITCH", `${body.head_pose.pitch.toFixed(1)}°`],
    ["ROLL", `${body.head_pose.roll.toFixed(1)}°`],
    ["TIME", `${body.processing_time_seconds.toFixed(2)}s`],
  ]);
  showImage(resultImage, resultFrame, `data:image/png;base64,${body.annotated_image_png_base64}`);
  logActivity("01 DETECT", `${body.landmark_count} landmarks, ${body.processing_time_seconds.toFixed(2)}s`);
});

const swapForm = document.getElementById("swap-form");
const swapStatusEl = document.getElementById("swap-status");
const progressSwap = document.getElementById("progress-swap");
const swapReadoutEl = document.getElementById("swap-readout");
const swapOriginalFrame = document.getElementById("swap-original-frame");
const swapOriginalImage = document.getElementById("swap-original-image");
const swapResultFrame = document.getElementById("swap-result-frame");
const swapResultImage = document.getElementById("swap-result-image");
let swapOriginalImageObjectUrl = null;

attachMagnifier(swapOriginalFrame, swapOriginalImage);
attachMagnifier(swapResultFrame, swapResultImage);

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

  swapStatusEl.textContent = "Processing...";
  progressSwap.hidden = false;
  setReadout(swapReadoutEl, []);
  swapResultImage.hidden = true;
  swapResultFrame.classList.remove("has-image");

  let response;
  let body;
  try {
    response = await fetch("/swap", { method: "POST", body: formData });
    body = await response.json();
  } catch (err) {
    swapStatusEl.textContent = "Error: unexpected response from server.";
    logActivity("02 SWAP", "Could not read server response", true);
    progressSwap.hidden = true;
    return;
  }

  progressSwap.hidden = true;

  if (!response.ok) {
    swapStatusEl.textContent = `Error: ${body.error}`;
    logActivity("02 SWAP", `Error: ${body.error}`, true);
    return;
  }

  const similarityText = body.identity_similarity === null
    ? "n/a"
    : `${(body.identity_similarity * 100).toFixed(1)}%`;
  swapStatusEl.textContent = "Face swap complete.";
  setReadout(swapReadoutEl, [
    ["SIMILARITY", similarityText],
    ["TIME", `${body.processing_time_seconds.toFixed(2)}s`],
  ]);
  showImage(swapResultImage, swapResultFrame, `data:image/png;base64,${body.swapped_image_png_base64}`);
  logActivity("02 SWAP", `${similarityText}, ${body.processing_time_seconds.toFixed(2)}s`);
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

const lipsyncForm = document.getElementById("lipsync-form");
const lipsyncStatusEl = document.getElementById("lipsync-status");
const progressLipsync = document.getElementById("progress-lipsync");
const lipsyncReadoutEl = document.getElementById("lipsync-readout");
const lipsyncOriginalFrame = document.getElementById("lipsync-original-frame");
const lipsyncOriginalVideo = document.getElementById("lipsync-original-video");
const lipsyncResultFrame = document.getElementById("lipsync-result-frame");
const lipsyncResultVideo = document.getElementById("lipsync-result-video");
let lipsyncOriginalObjectUrl = null;

function showVideo(videoEl, frameEl, src) {
  videoEl.src = src;
  videoEl.hidden = false;
  frameEl.classList.add("has-image");
}

lipsyncForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const videoInput = document.getElementById("lipsync-video-input");
  const audioInput = document.getElementById("lipsync-audio-input");
  if (!videoInput.files.length || !audioInput.files.length) {
    return;
  }
  const videoFile = videoInput.files[0];

  if (lipsyncOriginalObjectUrl) {
    URL.revokeObjectURL(lipsyncOriginalObjectUrl);
  }
  lipsyncOriginalObjectUrl = URL.createObjectURL(videoFile);
  showVideo(lipsyncOriginalVideo, lipsyncOriginalFrame, lipsyncOriginalObjectUrl);

  const formData = new FormData();
  formData.append("video", videoFile);
  formData.append("audio", audioInput.files[0]);

  lipsyncStatusEl.textContent = "Processing... this may take a while.";
  progressLipsync.hidden = false;
  setReadout(lipsyncReadoutEl, []);
  lipsyncResultVideo.hidden = true;
  lipsyncResultFrame.classList.remove("has-image");

  let response;
  let body;
  try {
    response = await fetch("/lipsync", { method: "POST", body: formData });
    body = await response.json();
  } catch (err) {
    lipsyncStatusEl.textContent = "Error: unexpected response from server.";
    logActivity("03 LIP SYNC", "Could not read server response", true);
    progressLipsync.hidden = true;
    return;
  }

  progressLipsync.hidden = true;

  if (!response.ok) {
    lipsyncStatusEl.textContent = `Error: ${body.error}`;
    logActivity("03 LIP SYNC", `Error: ${body.error}`, true);
    return;
  }

  lipsyncStatusEl.textContent = "Lip sync complete.";
  setReadout(lipsyncReadoutEl, [
    ["TIME", `${body.processing_time_seconds.toFixed(2)}s`],
  ]);
  showVideo(lipsyncResultVideo, lipsyncResultFrame, body.result_url);
  logActivity("03 LIP SYNC", `${body.processing_time_seconds.toFixed(2)}s`);
});
