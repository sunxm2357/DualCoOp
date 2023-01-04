import numpy as np
import torch
import os
import shutil


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def average_precision(output, target):
    epsilon = 1e-8

    # sort examples
    indices = output.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(output), 1)))

    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i


def mAP(targs, preds):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """

    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))
    # compute average precision for each class
    for k in range(preds.shape[1]):
        # sort scores
        scores = preds[:, k]
        targets = targs[:, k]
        # compute average precision
        ap[k] = average_precision(scores, targets)
    return 100 * ap.mean()


def save_checkpoint(state, is_best, filepath='', prefix=None):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        if prefix is None:
            shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))
        else:
            shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, '%s_model_best.pth.tar' % prefix))


def add_weight_decay(model, weight_decay=1e-4, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def filter_samples(output, target, cls_id):
    target = target[:, cls_id]
    output = output[:, cls_id]
    non_zero_mask = torch.where(target.sum(-1))[0]
    target = target[non_zero_mask]
    output = output[non_zero_mask]
    return output, target


def one_hot_to_class_labels(one_hot_array):
    samples = []
    if isinstance(one_hot_array, np.ndarray):
        for i, s in enumerate(one_hot_array):
            idx_hot = np.where(s)[0]
            samples.append(list(idx_hot))
    return samples


def calc_F1(gt_labels, idxs, k, num_classes=None):
    TP = np.zeros(num_classes)
    FP = np.zeros(num_classes)
    class_samples = np.zeros(num_classes)
    gt_labels = one_hot_to_class_labels(gt_labels)

    num_samples = len(gt_labels)

    for i in range(num_samples):
        gt_label = gt_labels[i]
        if isinstance(gt_label, list):
            tps = [elem in idxs[i][:k] for elem in gt_label]
            for j in range(len(gt_label)):
                TP[gt_label[j]] += tps[j]
                class_samples[gt_label[j]] += 1
            fps = [elem not in gt_label for elem in idxs[i][:k]]

            for j in range(k):
                if j < FP.shape[0]:
                    FP[idxs[i][j]] += fps[j]
        else:
            raise NotImplementedError

    TP_s = np.nansum(TP)
    FP_s = np.nansum(FP)
    precision_o = TP_s / (TP_s + FP_s)

    class_samples_s = np.nansum(class_samples)
    recall_o = TP_s / class_samples_s

    if precision_o == 0 or recall_o == 0:  # avoid nan if both zero
        F1_o = 0
    else:
        F1_o = 2 * precision_o * recall_o / (precision_o + recall_o)

    return 100*precision_o, 100*recall_o, 100*F1_o