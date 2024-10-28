import numpy as np
import pandas as pd
import os
import sys
sys.path.append('../src')
from metainfo import img_dataset_list, txt_dataset_list, img_zs_model_list, txt_zs_model_list, full_dataset_list
from label_shift_utils import get_uniformly_resampled_indices, get_dirichlet_marginal, get_resampled_indices

def load_emb(dataset_name, model_name, base_path, normalize=True):
    """
    Load embeddings for a given dataset and model from base_path
    Normalize embeddings by default by dividing by L2 norm
    """
    if dataset_name not in full_dataset_list:
        raise Exception(f"{dataset_name} does not exist!" )
    if model_name not in img_zs_model_list + txt_zs_model_list:
        raise Exception(f"{model_name} does not exist!" )
    
    if model_name in txt_zs_model_list:
        emb = np.load(os.path.join(base_path, f"{dataset_name}_{model_name.replace('/', '')}_text_emb.npy"))
    else:
        emb = np.load(os.path.join(base_path, f"{dataset_name}_{model_name.replace('/', '')}_img_emb.npy"))
    
    if normalize:
        emb /= np.linalg.norm(emb, axis=-1, keepdims=True)
    return emb

def load_label_encoding(dataset_name, model_name, base_path, normalize=True):
    """
    Load label encodings (embeddings) for a given dataset and model from base_path
    """
    if dataset_name not in full_dataset_list:
        raise Exception(f"{dataset_name} does not exist!" )
    if model_name not in img_zs_model_list + txt_zs_model_list:
        raise Exception(f"{model_name} does not exist!" )
    
    label_encodings = np.load(os.path.join(base_path, f"{dataset_name}_{model_name.replace('/', '')}_y_emb.npy"))
    
    if normalize:
        label_encodings /= np.linalg.norm(label_encodings, axis=-1, keepdims=True)
    
    return label_encodings

def load_label(dataset_name, base_path, model_name=None):
    """
    Load labels for a given dataset.
    """
    if dataset_name not in full_dataset_list:
        raise Exception(f"{dataset_name} does not exist!" )
    
    if model_name in txt_zs_model_list:
        labels = np.load(os.path.join(base_path, f"{dataset_name}_{model_name.replace('/', '')}_y.npy"),
                        allow_pickle=True)
        labels = np.array(labels, dtype=int)
    else:
        labels = np.load(os.path.join(base_path, f"{dataset_name}_y.npy"))
        
    return labels


def load_top_label_encoding(dataset_name, model_name, base_path, normalize=True):
    """
    Load top label encodings for a given dataset and model from base_path (H-OTTER)
    """
    if dataset_name not in full_dataset_list:
        raise(f"{dataset_name} does not exist!" )
    if model_name not in img_zs_model_list + txt_zs_model_list:
        raise(f"{model_name} does not exist!" )
    
    label_encodings = np.load(os.path.join(base_path, f"{dataset_name}_{model_name.replace('/', '')}_top_y_emb.npy"))
    
    if normalize:
        label_encodings /= np.linalg.norm(label_encodings, axis=-1, keepdims=True)
    
    return label_encodings

def load_top_label(dataset_name, base_path, model_name=None):
    """
    Load top-level labels for a given dataset. (H-OTTER)
    """
    if dataset_name not in full_dataset_list:
        raise(f"{dataset_name} does not exist!" )
    
    if model_name in txt_zs_model_list:
        labels = np.load(os.path.join(base_path, f"{dataset_name}_{model_name.replace('/', '')}_top_y.npy"),
                        allow_pickle=True)
        labels = np.array(labels, dtype=int)
    else:
        labels = np.load(os.path.join(base_path, f"{dataset_name}_top_y.npy"))
        
    return labels

def get_n_classes(labels):
    """
    Get number of classes in a dataset
    """
    # Assume at least one label exist
    return len(pd.unique(labels))
    
def get_class_balance(labels, n_classes):
    """
    Get class balance for a dataset
    """

    save = {}
    for i in range(n_classes):
        save[i] = 0
    
    for l in labels:
        save[l] += 1
    
    class_balance = np.zeros(n_classes)
    
    for i in range(n_classes):
        class_balance[i] = save[i]
    class_balance /= len(labels)
    
    return class_balance

def get_fewshot_samples(train_emb, train_labels, n_classes, n_samples_per_class, seed, balanced_train=True):
    """
    Get few-shot samples from a training set
    """
    if balanced_train:
        sampled_indices_train = get_uniformly_resampled_indices(train_labels, n_classes, n_samples_per_class, seed)
    else:
        sampled_indices_train = np.random.choice(train_emb.shape[0], size=n_samples_per_class * n_classes, replace=False)

    train_emb_sampled = train_emb[sampled_indices_train]
    train_labels_sampled = train_labels[sampled_indices_train]
    
    return train_emb_sampled, train_labels_sampled

def get_label_shifted_samples(test_emb, test_labels, alpha, n_classes, seed):
    """
    Get label-shifted samples from a test set
    """
    target_cb = get_dirichlet_marginal(get_class_balance(test_labels, n_classes) * n_classes * alpha, seed)
    sampled_indices_test = get_resampled_indices(test_labels, num_labels=n_classes, Py=target_cb, seed=seed)

    test_emb_sampled = test_emb[sampled_indices_test]
    test_labels_sampled = test_labels[sampled_indices_test]
    
    return test_emb_sampled, test_labels_sampled
    
def compute_tv(cb1, cb2):
    """
    Compute total variation between two class balance
    """
    return np.abs(cb1 - cb2).sum()/2
