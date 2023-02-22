import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt

# device = 'cpu'

class SurrogateHeaviside(torch.autograd.Function):
    
    # Activation function with surrogate gradient
#     sigma = 100.0

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
    def __init__(self, threshold_fn, device, t=0.5, sigma=100., num_labels=10):
        super(ThresholdModel, self).__init__()
        
        # define num_labels seuils differents, initialisés à 0.5

#         self.dense = torch.nn.Linear(10, 10)

        self.thresh = torch.nn.Parameter(t*torch.ones(num_labels), requires_grad=True)
        self.sigma = torch.nn.Parameter(sigma*torch.ones(num_labels), requires_grad=True)
        self.threshold_fn = threshold_fn
        self.device = device
        
    
    def forward(self, x):
        out = self.threshold_fn(x.to(self.device, dtype=torch.float)-self.thresh.to(self.device, dtype=torch.float), 
                                self.sigma.to(self.device, dtype=torch.float))
#         out = out.clamp_(min=0.01, max=0.99)
        # out = self.dense(x.to(device, dtype=torch.float))
        # out = F.sigmoid(out)
        # out = self.threshold_fn(out-F.sigmoid(self.thresh.to(device, dtype=torch.float)))
        return out

    
    def clamp(self):
        
        self.thresh.data.clamp_(min=0., max=1.)

        
def F1_loss_objective(binarized_output, y_true):
    # let's first convert binary vector prob into logits
#     prob = torch.clamp(prob, 1.e-12, 0.9999999)
    
#     average = 'macro'
    average = 'micro'
    epsilon = torch.tensor(1e-12)
    
    if average == 'micro':
        y_true = torch.flatten(y_true)
        binarized_output = torch.flatten(binarized_output)
        
    true_positives = torch.sum(y_true * binarized_output, dim=0)
    predicted_positives = torch.sum(binarized_output, dim=0)
    positives = torch.sum(y_true, dim=0)
    precision = true_positives / (predicted_positives + epsilon)
    recall = true_positives / (positives + epsilon)

    f1 = 2 * ((precision * recall) / (precision + recall + epsilon))
#     return precision, recall, f1
    return - f1.mean()


def train_thresholding_model(model: ThresholdModel, predictions, labels, epochs: int, criterion, num_labels: int, lr=0.00001, verbose=True):
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
        if epoch % 10 == 0: print('threshs:', learned_AT_thresholds)
        # if torch.sum(delta_thresh) < 0.01: break
    print('-'*20)
    plt.figure()
    # plt.figure(figsize=(8,6))
    plt.plot([loss.detach().cpu() for loss in losses])