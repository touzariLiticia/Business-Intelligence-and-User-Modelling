import math
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from torch.autograd import Variable
import tkinter
import matplotlib.pyplot as plt 

def bin_initializer(num_bins=10):
    """
    Initialize a dict of bins, each bin is a dict with it's length, accuracy and confidence 
    """
    bin_dict = {}
    for i in range(num_bins):
        bin_dict[i] = {}
        bin_dict[i]['count'] = 0 # number of samples in the bin 
        bin_dict[i]['conf'] = 0 # sum of the bin's samples confidence 
        bin_dict[i]['acc'] = 0 # sum of the bin's samples accuracy 
        bin_dict[i]['bin_acc'] = 0 # average accuracy 
        bin_dict[i]['bin_conf'] = 0 # average confidence 
    return bin_dict  

def populate_bins(confs, preds, labels, num_bins=10):
    """
    Compute accuracy and confidence for each bin

    :confs: list
            liste of predicted probabilities (confidences)
    :preds: list 
            liste of the predicted labels
    :labels: list)
            liste of the correct labels
    :num_bins: int
            number of bins
    :return: dict
    """ 

    bin_dict = bin_initializer(num_bins)
    num_test_samples = len(confs)

    for i in range(0, num_test_samples):
        confidence = confs[i]
        prediction = preds[i]
        label = labels[i]
        # find the right bin for sample i
        binn = int(math.ceil(((num_bins * confidence) - 1)))
        bin_dict[binn]['count'] += 1
        bin_dict[binn]['conf'] += confidence
        bin_dict[binn]['acc'] += (1 if (label == prediction) else 0)

    for binn in range(0, num_bins):
        if (bin_dict[binn]['count'] > 0):
            bin_dict[binn]['bin_acc'] = bin_dict[binn]['acc']/\
                                        bin_dict[binn]['count']
            bin_dict[binn]['bin_conf'] = bin_dict[binn]['conf']/\
                                        bin_dict[binn]['count']
    return bin_dict

def expected_calibration_error(confs, preds, labels, num_bins=10):
    """
    Compute the ece loss 
    
    :confs: list
            liste of predicted probabilities (confidences)
    :preds: list 
            liste of the predicted labels
    :labels: list)
            liste of the correct labels
    :num_bins: int
            number of bins
    :return: float
            ece loss
    """ 
    
    bin_dict = populate_bins(confs, preds, labels, num_bins)
    num_samples = len(labels)
    ece = 0
    for i in range(num_bins):
        ece += (bin_dict[i]['count'] / num_samples) * \
        abs(bin_dict[i]['bin_acc'] - bin_dict[i]['bin_conf'])
    return ece

def maximum_calibration_error(confs, preds, labels, num_bins=10):
    """
    Compute the mce loss 
    
    :confs: list
            liste of predicted probabilities (confidences)
    :preds: list 
            liste of the predicted labels
    :labels: list)
            liste of the correct labels
    :num_bins: int
            number of bins
    :return: float
            mce loss
    """ 
    bin_dict = populate_bins(confs, preds, labels, num_bins)
    ce = []
    for i in range(num_bins):
        ce.append(abs(bin_dict[i]['bin_acc'] - bin_dict[i]['bin_conf']))
    return max(ce)

def average_calibration_error(confs, preds, labels, num_bins=10):
    """
    Compute the average calibration error 
    
    :confs: list
            liste of predicted probabilities (confidences)
    :preds: list 
            liste of the predicted labels
    :labels: list)
            liste of the correct labels
    :num_bins: int
            number of bins
    :return: float
            average calibration error 
    """ 
    bin_dict = populate_bins(confs, preds, labels, num_bins)
    non_empty_bins = 0
    ace = 0
    for i in range(num_bins):
        if bin_dict[i]['count'] > 0:
            non_empty_bins += 1
        ace += abs(bin_dict[i]['bin_acc'] - bin_dict[i]['bin_conf'])
    return ace / float(non_empty_bins)

def test_classification_net_logits(logits, labels):
    '''
    This function reports classification accuracy and confusion matrix given logits and labels
    from a model.

    :logits: Torch.tensor
            predicted probabilities
    :labels: Torch.tensor
            correct labels
    :return: confusion matrix
            
    '''
    labels_list = []
    predictions_list = []
    confidence_vals_list = []

    softmax = F.softmax(logits, dim=1)
    confidence_vals, predictions = torch.max(softmax, dim=1)
    labels_list.extend(labels.cpu().numpy().tolist())
    predictions_list.extend(predictions.cpu().numpy().tolist())
    confidence_vals_list.extend(confidence_vals.cpu().numpy().tolist())
    accuracy = accuracy_score(labels_list, predictions_list)
    return confusion_matrix(labels_list, predictions_list), accuracy, labels_list,\
        predictions_list, confidence_vals_list

def reliability_plot(confs, preds, labels, num_bins=15):
    '''
    Method to draw a reliability plot from a model's predictions and confidences.

    :confs: list
            liste of predicted probabilities (confidences)
    :preds: list 
            liste of the predicted labels
    :labels: list)
            liste of the correct labels
    :num_bins: int
            number of bins
    '''
    bin_dict = populate_bins(confs, preds, labels, num_bins)
    bns = [(i / float(num_bins)) for i in range(num_bins)]
    y = []
    for i in range(num_bins):
        y.append(bin_dict[i]['bin_acc'])
    plt.figure(figsize=(10, 8))  # width:20, height:3
    plt.bar(bns, bns, align='edge', width=0.06, color='mistyrose', label='Expected')
    plt.bar(bns, y, align='edge', width=0.05,
            color='blue', alpha=0.5, label='Actual')
    plt.ylabel('Accuracy')
    plt.xlabel('Confidence')
    plt.legend()
    plt.show()

def model_reliability_plot(model, valid_loader, device):
    
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.to(device)
                logits = model(input).to(device)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()
            softmaxes = F.softmax(logits, dim=1)
            confidences, predictions = torch.max(softmaxes, 1)
            
        reliability_plot(confidences, predictions, labels, num_bins=10)

class ECELoss(nn.Module):
    '''
    Compute ECE (Expected Calibration Error)
    '''
    def __init__(self, n_bins=10):
        super(ECELoss, self).__init__()
        self.bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        
    def forward(self, logits, labels):
        
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_boundaries[:-1], self.bin_boundaries[1:]):
            # Compute |confidence - accuracy| * (#bin / #samples) for each bin
            
            # Find the samples that belong to the bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            
            # If the bin isn't empty
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean() # Bin accuracy 
                avg_confidence_in_bin = confidences[in_bin].mean() # Bin confidence
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
        return ece

class AdaptiveECELoss(nn.Module):
    '''
    Compute Adaptive ECE
    '''
    def __init__(self, n_bins=15):
        super(AdaptiveECELoss, self).__init__()
        self.n_bins = n_bins

    def histedges_equalN(self, x):
        """
        This function returns the bins boundaries so as all bins 
        have the same number of samples
        """
        npt = len(x)
        return np.interp(np.linspace(0, npt, self.n_bins + 1),
                     np.arange(npt),
                     np.sort(x))
    def forward(self, logits, labels):
        
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
        
        n, bin_boundaries = np.histogram(confidences.cpu().detach(), self.histedges_equalN(confidences.cpu().detach()))
        
        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
            # Compute |confidence - accuracy| * (#bin / #samples) in each bin
            
            # Find the samples that belong to the bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            
            # If the bin isn't empty
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean() # Bin accuracy 
                avg_confidence_in_bin = confidences[in_bin].mean() # Bin confidence
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
        return ece

class ClasswiseECELoss(nn.Module):
    '''
    Compute Classwise ECE
    '''
    def __init__(self, num_classes, n_bins=10):
        super(ClasswiseECELoss, self).__init__()
        self.bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.num_classes = num_classes

    def forward(self, logits, labels):
        #num_classes = int((torch.max(labels) + 1).item())
        softmaxes = F.softmax(logits, dim=1)
        per_class_sce = None

        for i in range(self.num_classes):
            class_confidences = softmaxes[:, i]# The probability of each sample to belong to class i
            class_sce = torch.zeros(1, device=logits.device)
            labels_in_class = labels.eq(i) # one-hot vector of all positions where the label belongs to the class i

            for bin_lower, bin_upper in zip(self.bin_boundaries[:-1], self.bin_boundaries[1:]):
                in_bin = class_confidences.gt(bin_lower.item()) * class_confidences.le(bin_upper.item())
                prop_in_bin = in_bin.float().mean()
                if prop_in_bin.item() > 0:
                    accuracy_in_bin = labels_in_class[in_bin].float().mean()
                    avg_confidence_in_bin = class_confidences[in_bin].mean()
                    class_sce += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            if (i == 0):
                per_class_sce = class_sce
            else:
                per_class_sce = torch.cat((per_class_sce, class_sce), dim=0)

        sce = torch.mean(per_class_sce).view(-1)
        return sce



class FocalLoss(nn.Module):
    '''
    Compute Focall loss
    '''
    def __init__(self, gamma=0, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        return loss.mean().view(-1)


        










