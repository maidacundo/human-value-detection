import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
import torch
from torchmetrics import Accuracy
import torch.nn as nn

def print_classification_report_thresholding(true_labels, pred_labels, labels_columns, threshold, num_labels=20):
    pred_labels_threshold = np.where(pred_labels > threshold, 1, 0)

    classification_report = classification_report(
      true_labels, 
      pred_labels_threshold, 
      target_names=labels_columns, 
      zero_division=0
    )

    accuracy = Accuracy(task="multiclass", num_classes=num_labels)
    accuracy_value = accuracy(torch.tensor(pred_labels_threshold), torch.tensor(true_labels)).item()
    print(classification_report)
    print(f'accuracy: {accuracy_value}')
    return classification_report

def get_f1_optimized_thresholding(true_labels, pred_labels, labels_columns):
    threshold_list = []
    labels_results = {}
    for label in labels_columns:
        labels_results[label] = []

    for threshold in range(0, 90, 5):
        threshold = threshold / 100
        threshold_list.append(threshold)

        pred_labels_threshold = np.where(pred_labels > threshold, 1, 0)

        report = classification_report(
              true_labels, 
              pred_labels_threshold, 
              target_names=labels_columns, 
              zero_division=0,
              output_dict=True
            )

        for label in labels_columns:
            f1 = report[label]['f1-score']
            labels_results[label].append(f1)
    f1_score_optimized_thresholds = []

    for label in labels_columns:
        max_f1_idx = np.argmax(labels_results[label])
        f1_score_optimized_thresholds.append(threshold_list[max_f1_idx])
    return f1_score_optimized_thresholds

    
class SurrogateHeaviside(torch.autograd.Function):
    
    @staticmethod 
    def forward(ctx, input, sigma):
        
        output = torch.zeros_like(input)
        output[input > 0] = 1.0
        ctx.save_for_backward(input, sigma)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, sigma = ctx.saved_tensors
        grad_input = grad_output.clone()
        # approximation of the gradient using sigmoid function
        grad = grad_input*torch.sigmoid(sigma*input)*torch.sigmoid(-sigma*input)
        
        grad_sigma = grad_input*input*torch.sigmoid(sigma*input)*torch.sigmoid(-sigma*input)
        
        return grad, grad_sigma

threshold_fn = SurrogateHeaviside.apply


class ThresholdModel(nn.Module):
    def __init__(self, threshold_fn, device, t=0.5, sigma=100., num_labels=10, use_dense=False):
        super(ThresholdModel, self).__init__()

        self.dense = torch.nn.Linear(num_labels, num_labels)

        self.thresh = torch.nn.Parameter(t*torch.ones(num_labels), requires_grad=True)
        self.sigma = torch.nn.Parameter(sigma*torch.ones(num_labels), requires_grad=True)
        self.threshold_fn = threshold_fn
        self.device = device
        self.use_dense = use_dense
        
    
    def forward(self, x):
        if self.use_dense:
            out = self.dense(x.to(self.device, dtype=torch.float))
            out = torch.sigmoid(out.to(self.device, dtype=torch.float))
            out = self.threshold_fn(out.to(self.device, dtype=torch.float)-self.thresh.to(self.device, dtype=torch.float), 
                                    self.sigma.to(self.device, dtype=torch.float))
        else:
            out = self.threshold_fn(x.to(self.device, dtype=torch.float)-self.thresh.to(self.device, dtype=torch.float), 
                                self.sigma.to(self.device, dtype=torch.float))

        return out

def train_thresholding_model(model: ThresholdModel, predictions, labels, epochs: int, criterion, num_labels: int, lr=0.00001, verbose=True):

    predictions=torch.tensor(predictions, dtype=torch.float).to(model.device)
    labels=torch.tensor(labels, dtype=torch.float).to(model.device)
    model.to(model.device)
    
    cumul_delta_thresh = torch.zeros(num_labels,)
    delta_thresh = torch.zeros(num_labels,)
    
    for el in model.parameters():
        PREC_learned_AT_thresholds = el.clone().detach().cpu()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    for epoch in range(epochs):
                
        model.train()
        optimizer.zero_grad()

        outputs = model(predictions)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss)
        
        

        for el in model.parameters():
            learned_AT_thresholds = el.clone().detach().cpu()

        delta_thresh = learned_AT_thresholds - PREC_learned_AT_thresholds
        cumul_delta_thresh += delta_thresh
        PREC_learned_AT_thresholds = learned_AT_thresholds
        if verbose:
            print ('Epoch [{}], Loss: {:.4f}'.format(epoch+1, loss))
    print('-'*20)
    plt.figure()
    plt.plot([loss.detach().cpu() for loss in losses])