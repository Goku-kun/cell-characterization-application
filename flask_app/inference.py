import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings("ignore")


from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import segmenter
import classifier
    
def infer(imgname, img_dir):   

    imgpath = img_dir + imgname
    print(imgpath)
    cell_ids, cell_array = segmenter.segmenter(imgpath)

    eval_results, cell_counts, image = classifier.analyzer(imgname, img_dir, imgpath, cell_ids, cell_array) 

    return eval_results, cell_counts, image