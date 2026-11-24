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

  const response = await fetch("/detect", { method: "POST", body: formData });
  const body = await response.json();

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
