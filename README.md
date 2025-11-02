🧠 DementiaApp
AI-Powered MRI Dementia Stage Classification

DementiaApp is a Flask-based web application that predicts the stage of dementia from uploaded MRI brain scans.
It integrates a Hybrid Deep Learning Model combining ResNet18 (for feature extraction) and a Quantum-Inspired Neural Layer to achieve accurate, explainable predictions.

🚀 Features

Upload MRI scan images directly through the web interface.

Predicts one of four stages:

🟢 Non Demented

🟡 Very Mild Demented

🟠 Mild Demented

🔴 Moderate Demented

Displays prediction confidence.

Built with PyTorch, Flask, and Render Deployment.

🧩 Tech Stack

Backend: Python, Flask

Model: ResNet18 + Dummy Quantum Layer

Frontend: HTML, CSS (Jinja2 Templates)

Deployment: Render (Free Tier)

📁 Project Structure
DementiaApp/
├── app.py
├── hybrid_resnet_qnn_final.pth
├── requirements.txt
├── start.sh
├── runtime.txt
├── templates/
│     └── index.html
└── static/
      └── uploads/


🧠 Model Details

Architecture: ResNet18 feature extractor + custom Quantum-Inspired layer.

Classes: 4 Dementia stages.

Framework: PyTorch.

Model File: hybrid_resnet_qnn_final.pth (loaded automatically in app.py).

🧾 License

This project is open-source and available under the MIT License.

👩‍💻 Authors

Sameera Begum Shaik , Harini Sree Gundralla
📍 CSE (Business Systems & AIML), RGMCET
💡 Passionate about AI, ML, and Cognitive Health Solutions
