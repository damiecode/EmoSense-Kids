import os, json, pickle, warnings
import base64, io, pathlib

import numpy as np
import torch
from PIL import Image

import vgg
from cp_net import CPNet
from cp_util import DATA_TRANSFORMS


MODEL_PATH = pathlib.Path(__file__).parent/"prod_model/attempt-2-best.pth"

SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_MODEL = CPNet(num_class=6, pretrained=False).to(DEVICE)
IMAGE_MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
IMAGE_MODEL.eval()


def pain_emotion_from_image(body):
    class_to_idx = {
                '0': f"1 - No pain", '10': "10 - Worst Possible Pain",
                '2': "2 - Mild pain", '4': "4 - Moderate Pain",
                '6': "6 - Severe Pain", '8': "8 - Very Severe Pain"
            }
    if body:
        b64_encoded_img = body["image"]
        image = Image.open(io.BytesIO(base64.b64decode(b64_encoded_img))).convert("RGB")
        image_t = DATA_TRANSFORMS["test"](image).unsqueeze(0).to(DEVICE)
        output = IMAGE_MODEL(image_t)
        output_probs = torch.nn.functional.softmax(output, dim=1)
        idx_to_class = {0: '0', 1: '10', 2: '2', 3: '4', 4: '6', 5: '8'}
        probs, classes = torch.topk(output_probs, 5)
        classes = [class_to_idx[idx_to_class[i]] for i in classes.squeeze().tolist()]
    
        return {
                "predicted_classes": classes,
                "probabilities":probs.squeeze().tolist()
            }

    else:
        return {}
