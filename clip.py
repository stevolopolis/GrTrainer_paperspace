import torch
import torchmetrics
import cv2

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from parameters import Params
from data_loader_v2 import DataLoader
from evaluation import visualize_cls, denormalize_img

from transformers import CLIPProcessor, CLIPModel

params = Params()
LABELS = ["Chair", "Lamp", "Figurine", "Plants", "Pen"]

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
model.to(params.DEVICE)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

data_loader = DataLoader(params.TRAIN_PATH,
                        1,
                        train_val_split=params.TRAIN_VAL_SPLIT,
                        include_depth=False,
                        verbose=False,
                        seed=42,
                        device=params.DEVICE)

i = 0
total = data_loader.n_data
pred_ls = []
gt_ls = []

pbar = tqdm(data_loader.load_cls(include_depth=False), total=total)
for (img, cls_map, label) in pbar:
    inputs = processor(text=LABELS, images=img[0], return_tensors="pt", padding=True)
    inputs.to(params.DEVICE)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)

    pred_ls.append(torch.argmax(probs))
    gt_ls.append(label)

    i += 1

    if (i % 50 == 0):
        accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=5)
        acc = accuracy(torch.tensor(pred_ls), torch.tensor(gt_ls))
        confmat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=5, normalize='true')
        mat = confmat(torch.tensor(pred_ls), torch.tensor(gt_ls))

        plt.matshow(mat)
        plt.title('Problem 1: Confusion Matrix Digit Recognition')
        plt.colorbar()
        plt.ylabel('True Label')
        plt.xlabel('Predicated Label')
        plt.savefig('confusion_matrix.jpg')
        pbar.set_description("Acc: %s" % acc)
    

    """print(LABELS[torch.argmax(probs)], LABELS[label])
    img_bgr = denormalize_img(img)
    img_vis = np.ascontiguousarray(img_bgr, dtype=np.uint8)
    cv2.imshow('vis', img_vis)
    cv2.waitKey(0)"""
