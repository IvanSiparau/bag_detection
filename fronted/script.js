async function uploadVideo() {
    const fileInput = document.getElementById("videoUpload");
    const status = document.getElementById("status");

    if (!fileInput.files.length) {
        alert("Выберите видео!");
        return;
    }

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append("file", file);

    status.textContent = "Идет обработка видео...";

    try {
        const response = await fetch("http://127.0.0.1:8000/upload", {
            method: "POST",
            body: formData
        });
        const data = await response.json();
        status.textContent = data.message;
    } catch (err) {
        status.textContent = "Ошибка при загрузке видео";
        console.error(err);
    }
}

function downloadVideo() {
    window.location.href = "http://127.0.0.1:8000/download";
}
