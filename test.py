from __future__ import print_function

import time
import init_paths
import skimage.io as skio
import os
import demo
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


def f1(output, mask, eps=1e-6):
    # <-- the name image is not professional
    # <-- img need to be rounded before calculating
    output = output.round()
    TP = np.sum(output * mask)
    FP = np.sum(output * (1.0 - mask))
    FN = np.sum((1.0 - output) * mask)
    f1_val = (2*TP + eps) / (2*TP + FN + FP + eps)
    return f1_val

ckpt_path = './ckpt/exif_final/exif_final.ckpt'
exif_demo = demo.Demo(ckpt_path=ckpt_path, use_gpu=[0], quality=2.0, num_per_dim=10)
path_dataset = "dataset"
datasets = ["RT", "Columbia"]  # <-- it should be datasets = [...]
for dataset in datasets:       # <-- it should be for dataset in datasets
    # Collect paths
    # au_path_images = os.path.join(path_dataset, dataset, "au/images")
    # au_images = [os.path.join(au_path_images, f ) for f in os.listdir(au_path_images)]

    # tp_path_images = os.path.join(path_dataset, dataset, "tp/images")
    # tp_images = [os.path.join(tp_path_images, f ) for f in os.listdir(tp_path_images)]

    # masks_path = os.path.join(path_dataset, dataset, "tp/labels")
    # masks = [os.path.join(masks_path, f ) for f in os.listdir(masks_path)]      # <-- Ensure that tp_images and masks are in the same order

    # # Alternative for collecting paths
    au_images = sorted(glob(os.path.join(path_dataset, dataset, "au/images/*.*")))
    tp_images = sorted(glob(os.path.join(path_dataset, dataset, "tp/images/*.*")))
    masks = sorted(glob(os.path.join(path_dataset, dataset, "tp/labels/*.*")))


    f1_score = 0
    print(dataset)
    print("Num of tp images ", len(tp_images))
    # for j in tqdm(range(len(tp_images)), total=len(tp_images)):
    #     img = cv2.imread(tp_images[j])[...,::-1]  # <-- This is BGR, not RGB. For reading RGB: img = cv2.imread(tp_images[j])[...,::-1]
    #     img = cv2.resize(img, (256, 256))
    #     star_time = time.time()
    #     output = exif_demo.run(img, use_ncuts=True, blue_high=True)
    #     end_time = time.time()

    #     mask = cv2.imread(masks[j], 0)
    #     mask = cv2.resize(mask, (256, 256))
    #     mask[mask>0]=1
    #     mask = mask.astype(np.float32)

    #     score1 = f1(output, mask)  
    #     score2 = f1(output, 1-mask)     
    #     score = score1 if score1>score2 else score2
    #     f1_score += score
    # print("F1 score for tp images ", f1_score / len(tp_images))


    print("Num of au images ", len(au_images))
    for image in tqdm(au_images, total=len(au_images)):
        img = cv2.imread(image)[...,::-1]   # <-- This is BGR, not RGB. For reading RGB: img = img = cv2.imread(image)[...,::-1]
        img = cv2.resize(img, (256, 256))
        output = exif_demo.run(img, use_ncuts=True, blue_high=True)     # <-- the name image is not professional
        mask = np.zeros((256, 256)).astype(np.float32) # <-- default of np.zeros is np.float64, not np.float32, must be astype(np.float32)

        score = f1(output, mask)  
        f1_score += score
#     print("F1 score for both ", f1_score / (len(tp_images) + len(au_images)))
    print("F1 score for au ", f1_score / len(au_images))
