function uploadImage() {
    let fileInput = document.getElementById("fileInput");
    let file = fileInput.files[0];

    if (!file) {
        alert("Por favor, selecciona una imagen.");
        return;
    }

    let formData = new FormData();
    formData.append("file", file);

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("result").innerText = 
            `ResNet: ${data['ResNet Prediction']}\n` +
            `AlexNet: ${data['AlexNet Prediction']}`;
    })
    .catch(error => console.error("Error:", error));
}
