import os
from flask import Flask, render_template, request
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np

# ------------------------------
# Optimize Memory for Render
# ------------------------------
torch.set_num_threads(1)  # Prevent PyTorch from using multiple threads (saves memory)
device = torch.device("cpu")  # Force CPU on Render free tier

# ------------------------------
# Dummy Quantum Layer
# ------------------------------
class DummyQuantumLayer(nn.Module):
    def __init__(self, n_qubits):
        super().__init__()
        self.fc = nn.Linear(n_qubits, n_qubits)

    def forward(self, x):
        return torch.tanh(self.fc(x))  # simulate quantum nonlinearity


# ------------------------------
# Feature Extractor (ResNet18)
# ------------------------------
base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for param in base_model.parameters():
    param.requires_grad = False  # freeze pretrained weights

# Remove final FC layer and flatten
base_model = nn.Sequential(*list(base_model.children())[:-1], nn.Flatten())

# ------------------------------
# Classical Reducer + Quantum + Classifier
# ------------------------------
n_qubits = 4
num_classes = 4  # Mild, Moderate, Non, Very Mild

classical_reducer = nn.Linear(512, n_qubits)
quantum_layer = DummyQuantumLayer(n_qubits)

# ------------------------------
# Hybrid Model Definition
# ------------------------------
class HybridResNetQNN(nn.Module):
    def __init__(self, base_model, reducer, qlayer, n_qubits, n_classes):
        super().__init__()
        self.feature = base_model
        self.reducer = reducer
        self.qlayer = qlayer
        self.classifier = nn.Linear(n_qubits, n_classes)

    def forward(self, x):
        feats = self.feature(x)          # (B,512)
        reduced = self.reducer(feats)    # (B, n_qubits)
        angles = (reduced + 1.0) * (np.pi / 2)  # scale [-1,1] → [0,π]
        q_out = self.qlayer(angles)
        out = self.classifier(q_out)
        return out


# ------------------------------
# Load Model (Render-friendly + Safe)
# ------------------------------
model = HybridResNetQNN(base_model, classical_reducer, quantum_layer, n_qubits, num_classes).to(device)
model_path = "hybrid_resnet_qnn_final.pth"

if os.path.exists(model_path):
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"✅ Model loaded successfully on CPU from {model_path}")
    except Exception as e:
        print(f"⚠️ Error loading model: {e}")
else:
    print(f"⚠️ Warning: model weights not found at {model_path}")

model.eval()


# ------------------------------
# Flask App Setup
# ------------------------------
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class_names = ["Mild Dementia", "Moderate Dementia", "Non Demented", "Very Mild Demented"]


# ------------------------------
# Routes
# ------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    if not file:
        return "No file uploaded!", 400

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    img = Image.open(path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)
        pred_idx = probs.argmax(1).item()
        confidence = round(probs[0][pred_idx].item() * 100, 2)

    return render_template("index.html",
                           prediction=class_names[pred_idx],
                           confidence=confidence,
                           image_path=path)


# ------------------------------
# Render-friendly App Run
# ------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render gives PORT dynamically
    app.run(host="0.0.0.0", port=port)
