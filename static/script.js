const form = document.getElementById("upload-form");
const statusEl = document.getElementById("status");
const resultImage = document.getElementById("result-image");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const fileInput = document.getElementById("image-input");
  if (!fileInput.files.length) {
    return;
  }

  const formData = new FormData();
  formData.append("image", fileInput.files[0]);

  statusEl.textContent = "İşleniyor...";
  resultImage.hidden = true;

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

  statusEl.textContent = `${body.landmark_count} landmark noktası bulundu.`;
  const pose = body.head_pose;
  statusEl.textContent += ` (Yaw: ${pose.yaw.toFixed(1)}°, Pitch: ${pose.pitch.toFixed(1)}°, Roll: ${pose.roll.toFixed(1)}°)`;
  resultImage.src = `data:image/png;base64,${body.annotated_image_png_base64}`;
  resultImage.hidden = false;
});

const swapForm = document.getElementById("swap-form");
const swapStatusEl = document.getElementById("swap-status");
const swapResultImage = document.getElementById("swap-result-image");

swapForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const sourceInput = document.getElementById("source-input");
  const targetInput = document.getElementById("target-input");
  if (!sourceInput.files.length || !targetInput.files.length) {
    return;
  }

  const formData = new FormData();
  formData.append("source", sourceInput.files[0]);
  formData.append("target", targetInput.files[0]);

  swapStatusEl.textContent = "İşleniyor...";
  swapResultImage.hidden = true;

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

  swapStatusEl.textContent = "Yüz değiştirme tamamlandı.";
  swapResultImage.src = `data:image/png;base64,${body.swapped_image_png_base64}`;
  swapResultImage.hidden = false;
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
