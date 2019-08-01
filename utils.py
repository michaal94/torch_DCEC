import numpy as np
import sklearn.metrics
import torch
from torchvision import transforms

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
inv_normalize = transforms.Normalize(
    mean=[-0.485 / .229, -0.456 / 0.224, -0.406 / 0.255],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
)


# Simple tensor to image translation
def tensor2img(tensor):
    img = tensor.cpu().data[0]
    if img.shape[0] != 1:
        img = inv_normalize(img)
    img = torch.clamp(img, 0, 1)
    return img


# Define printing to console and file
def print_both(f, text):
    print(text)
    f.write(text + '\n')


# Metrics class was copied from DCEC article authors repository (link in README)
class metrics:
    nmi = sklearn.metrics.normalized_mutual_info_score
    ari = sklearn.metrics.adjusted_rand_score

    @staticmethod
    def acc(labels_true, labels_pred):
        labels_true = labels_true.astype(np.int64)
        assert labels_pred.size == labels_true.size
        D = max(labels_pred.max(), labels_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(labels_pred.size):
            w[labels_pred[i], labels_true[i]] += 1
        from sklearn.utils.linear_assignment_ import linear_assignment
        ind = linear_assignment(w.max() - w)
        return sum([w[i, j] for i, j in ind]) * 1.0 / labels_pred.size
