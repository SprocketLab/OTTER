import numpy as np
import torch
import ot
import pandas as pd
import random
import sklearn

def get_pred_proba(emb, label_encodings, temperature=1):
    """
    Get prediction probabilities for a given embedding and label encodings
    """
    M = np.exp(sklearn.metrics.pairwise.cosine_similarity(emb, label_encodings)/temperature) # cosine similarity
    M = M / M.sum(axis=1, keepdims=1)
    
    return M

def zs_predict(emb, label_encodings, temperature=1):
    """
    Get predictions for a given embedding and label encodings
    """
    pred_proba = get_pred_proba(emb, label_encodings, temperature)
    return pred_proba.argmax(axis=1)
    

def nlh_cost_matrix(emb, label_encodings, epsilon=1e-9):
    """
    Get negative log-likelihood cost matrix for a given embedding and label encodings
    """
    M = get_pred_proba(emb, label_encodings)
    if epsilon is None:
        M = -np.log(M)
    else: # Prevent log0 exception
        M = -np.log(M + epsilon)
    return M

def ot_predict(emb, label_encodings, n_classes, class_balance, epsilon=1e-9):
    """
    Get predictions for a given embedding and label encodings using optimal transport
    """
    src = np.ones(emb.shape[0]) / emb.shape[0]
    tgt = np.ones(n_classes) * class_balance
    M = nlh_cost_matrix(emb, label_encodings, epsilon)
    T = ot.emd(src, tgt, M, numItermax=10000000)
    pred = T.argmax(axis=1)
    
    return pred

def ot_posthoc(pred_proba, n_classes, class_balance, epsilon=1e-6):
    """
    Get predictions for a given prediction probabilities using optimal transport
    """
    src = np.ones(pred_proba.shape[0]) / pred_proba.shape[0]
    tgt = np.ones(n_classes) * class_balance
    if epsilon is None:
        M = -np.log(pred_proba)
    else:
        M = -np.log(pred_proba + epsilon)
        
    T = ot.emd(src, tgt, M, numItermax=10000000)
    pred = T.argmax(axis=1)
    
    return pred

def ot_posthoc_logits(logits, n_classes, class_balance):
    """
    Get predictions for a given logits using optimal transport
    """
    src = np.ones(logits.shape[0]) / logits.shape[0]
    tgt = np.ones(n_classes) * class_balance
    M = -logits
    T = ot.emd(src, tgt, M, numItermax=10000000)
    pred = T.argmax(axis=1)
    
    return pred


def pm_predict(emb, label_encodings, n_classes, class_balance,
               base_temperature=1e-5, learning_rate=1e-5, loss_threshold=1e-12, max_iter=10000):
    """
    Get predictions for a given embedding and label encodings using prior matching
    """
    # temperature = base_temperature / n_classes
    temperature = base_temperature
    pred_proba = get_pred_proba(emb, label_encodings)
    pm_reweighter = Reweighter(n_inputs=pred_proba.shape[1], temperature=temperature).cuda()
    optimizer = torch.optim.AdamW(pm_reweighter.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()
    
    prev_loss = 1
    it = 0
    
    pred_proba = torch.tensor(pred_proba, dtype=torch.float32).cuda()
    class_balance = torch.tensor(class_balance, dtype=torch.float32).cuda()
    
    while prev_loss >= loss_threshold:
        optimizer.zero_grad()
        pred = pm_reweighter(pred_proba)
        pred_soft_class_balance = pred.mean(dim=0)
        loss = loss_fn(pred_soft_class_balance, class_balance)
        prev_loss = loss.item()
        loss.backward()
        optimizer.step()
        it += 1
#             if (it+1)%10==0:
#                 print(model_name, dataset_name, 'it', it+1, 'loss:', loss.item())
        if it >= max_iter:
            break
    rw_pred_proba = pm_reweighter(pred_proba)
    return rw_pred_proba.argmax(axis=-1).cpu().numpy()


def pm_posthoc(pred_proba, n_classes, class_balance,
               base_temperature=1e-5, learning_rate=1e-5, loss_threshold=1e-12, max_iter=10000):
    """
    Get predictions for a given prediction probabilities using prior matching
    """
    temperature = base_temperature
    pm_reweighter = Reweighter(n_inputs=pred_proba.shape[1], temperature=temperature).cuda()
    optimizer = torch.optim.AdamW(pm_reweighter.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()
    
    prev_loss = 1
    it = 0
    
    pred_proba = torch.tensor(pred_proba, dtype=torch.float32).cuda()
    class_balance = torch.tensor(class_balance, dtype=torch.float32).cuda()
    
    while prev_loss >= loss_threshold:
        optimizer.zero_grad()
        pred = pm_reweighter(pred_proba)
        pred_soft_class_balance = pred.mean(dim=0)
        loss = loss_fn(pred_soft_class_balance, class_balance)
        prev_loss = loss.item()
        loss.backward()
        optimizer.step()
        it += 1
#             if (it+1)%10==0:
#                 print(model_name, dataset_name, 'it', it+1, 'loss:', loss.item())
        if it >= max_iter:
            break
    rw_pred_proba = pm_reweighter(pred_proba)
    return rw_pred_proba.argmax(axis=-1).cpu().numpy()

def pm_posthoc_logits(logits, n_classes, class_balance,
               base_temperature=1e-5, learning_rate=1e-5, loss_threshold=1e-12, max_iter=10000):
    """
    Get predictions for a given logits using prior matching
    """
    sftmax = torch.nn.Softmax(dim=-1)
    pred_proba = sftmax(logits)
    temperature = base_temperature
    pm_reweighter = Reweighter(n_inputs=pred_proba.shape[1], temperature=temperature).cuda()
    optimizer = torch.optim.AdamW(pm_reweighter.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()
    
    prev_loss = 1
    it = 0
    
    pred_proba = torch.tensor(pred_proba, dtype=torch.float32).cuda()
    class_balance = torch.tensor(class_balance, dtype=torch.float32).cuda()
    
    while prev_loss >= loss_threshold:
        optimizer.zero_grad()
        pred = pm_reweighter(pred_proba)
        pred_soft_class_balance = pred.mean(dim=0)
        loss = loss_fn(pred_soft_class_balance, class_balance)
        prev_loss = loss.item()
        loss.backward()
        optimizer.step()
        it += 1
#             if (it+1)%10==0:
#                 print(model_name, dataset_name, 'it', it+1, 'loss:', loss.item())
        if it >= max_iter:
            break
    rw_pred_proba = pm_reweighter(pred_proba)
    return rw_pred_proba.argmax(axis=-1).cpu().numpy()
    
    


class Reweighter(torch.nn.Module):
    def __init__(self, n_inputs, temperature):
        super(Reweighter, self).__init__()
        random_init = torch.rand(n_inputs)
        random_init /= random_init.sum()
        self.weights = torch.nn.parameter.Parameter(random_init, requires_grad=True)
        self.sfmax = torch.nn.Softmax(dim=-1)
        self.temperature = temperature
    # make predictions
    def forward(self, x):
        y_pred = self.sfmax(self.weights * x / self.temperature)
        return y_pred
