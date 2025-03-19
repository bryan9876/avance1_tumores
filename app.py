import torch
import torchvision.models as models
import torchvision.transforms as transforms
import json
import urllib.request
from flask import Flask, request, jsonify, render_template
from PIL import Image

app = Flask(__name__)

# Descargar etiquetas de ImageNet
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
urllib.request.urlretrieve(url, "imagenet_classes.txt")

# Cargar las clases de ImageNet
with open("imagenet_classes.txt", "r") as f:
    imagenet_classes = [line.strip() for line in f]

# Cargar modelos preentrenados
resnet = models.resnet50(pretrained=True)
alexnet = models.alexnet(pretrained=True)

# Poner modelos en modo evaluación
resnet.eval()
alexnet.eval()

# Transformaciones para las imágenes
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    image = Image.open(file.stream)
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        resnet_output = resnet(image)
        alexnet_output = alexnet(image)

    resnet_pred_idx = torch.argmax(resnet_output, 1).item()
    alexnet_pred_idx = torch.argmax(alexnet_output, 1).item()

    return jsonify({
        'ResNet Prediction': imagenet_classes[resnet_pred_idx],
        'AlexNet Prediction': imagenet_classes[alexnet_pred_idx],
        'Model Performance': {
            "ResNet-50": {"Top-1 Accuracy": "76.2%", "Top-5 Accuracy": "92.9%"},
            "AlexNet": {"Top-1 Accuracy": "56.5%", "Top-5 Accuracy": "79.1%"}
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
