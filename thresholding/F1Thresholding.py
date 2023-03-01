import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt

class F1Thresholding:
    def __init__(self, true_labels_train, pred_labels_train):
        super().__init__()
        self.true_labels_train, self.pred_labels_train = true_labels_train, pred_labels_train
        
        