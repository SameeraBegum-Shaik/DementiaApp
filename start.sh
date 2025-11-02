#!/bin/bash

# If your model is already in the repo, no need to download.
# But if it's hosted externally, you can download it like this:
# wget "https://huggingface.co/yourname/model.pth" -O hybrid_resnet_qnn_final.pth

echo "🚀 Starting Dementia Detection Flask App..."
gunicorn app:app --bind 0.0.0.0:$PORT
