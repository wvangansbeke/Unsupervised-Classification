"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import numpy as np
import torch
import torch.nn.functional as F
from utils.common_config import get_feature_dimensions_backbone
from utils.utils import AverageMeter, confusion_matrix
from data.custom_dataset import NeighborsDataset
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from losses.losses import entropy


@torch.no_grad()
def contrastive_evaluate(val_loader, model, memory_bank):
    top1 = AverageMeter('Acc@1', ':6.2f')
    model.eval()

    for batch in val_loader:
        images = batch['image'].cuda(non_blocking=True)
        target = batch['target'].cuda(non_blocking=True)

        output = model(images)
        output = memory_bank.weighted_knn(output) 

        acc1 = 100*torch.mean(torch.eq(output, target).float())
        top1.update(acc1.item(), images.size(0))

    return top1.avg


@torch.no_grad()
def get_predictions(p, dataloader, model, return_features=False):
    # Make predictions on a dataset with neighbors
    model.eval()
    predictions = [[] for _ in range(p['num_heads'])]
    probs = [[] for _ in range(p['num_heads'])]
    targets = []
    if return_features:
        ft_dim = get_feature_dimensions_backbone(p)
        features = torch.zeros((len(dataloader.sampler), ft_dim)).cuda()
    
    if isinstance(dataloader.dataset, NeighborsDataset): # Also return the neighbors
        key_ = 'anchor'
        include_neighbors = True
        neighbors = []

    else:
        key_ = 'image'
        include_neighbors = False

    ptr = 0
    for batch in dataloader:
        images = batch[key_].cuda(non_blocking=True)
        bs = images.shape[0]
        res = model(images, forward_pass='return_all')
        output = res['output']
        if return_features:
            features[ptr: ptr+bs] = res['features']
            ptr += bs
        for i, output_i in enumerate(output):
            predictions[i].append(torch.argmax(output_i, dim=1))
            probs[i].append(F.softmax(output_i, dim=1))
        targets.append(batch['target'])
        if include_neighbors:
            neighbors.append(batch['possible_neighbors'])

    predictions = [torch.cat(pred_, dim = 0).cpu() for pred_ in predictions]
    probs = [torch.cat(prob_, dim=0).cpu() for prob_ in probs]
    targets = torch.cat(targets, dim=0)

    if include_neighbors:
        neighbors = torch.cat(neighbors, dim=0)
        out = [{'predictions': pred_, 'probabilities': prob_, 'targets': targets, 'neighbors': neighbors} for pred_, prob_ in zip(predictions, probs)]

    else:
        out = [{'predictions': pred_, 'probabilities': prob_, 'targets': targets} for pred_, prob_ in zip(predictions, probs)]

    if return_features:
        return out, features.cpu()
    else:
        return out


@torch.no_grad()
def scan_evaluate(predictions):
    # Evaluate model based on SCAN loss.
    num_heads = len(predictions)
    output = []

    for head in predictions:
        # Neighbors and anchors
        probs = head['probabilities']
        neighbors = head['neighbors']
        anchors = torch.arange(neighbors.size(0)).view(-1,1).expand_as(neighbors)

        # Entropy loss
        entropy_loss = entropy(torch.mean(probs, dim=0), input_as_probabilities=True).item()
        
        # Consistency loss       
        similarity = torch.matmul(probs, probs.t())
        neighbors = neighbors.contiguous().view(-1)
        anchors = anchors.contiguous().view(-1)
        similarity = similarity[anchors, neighbors]
        ones = torch.ones_like(similarity)
        consistency_loss = F.binary_cross_entropy(similarity, ones).item()
        
        # Total loss
        total_loss = - entropy_loss + consistency_loss
        
        output.append({'entropy': entropy_loss, 'consistency': consistency_loss, 'total_loss': total_loss})

    total_losses = [output_['total_loss'] for output_ in output]
    lowest_loss_head = np.argmin(total_losses)
    lowest_loss = np.min(total_losses)

    return {'scan': output, 'lowest_loss_head': lowest_loss_head, 'lowest_loss': lowest_loss}


@torch.no_grad()
def hungarian_evaluate(subhead_index, all_predictions, class_names=None, 
                        compute_purity=True, compute_confusion_matrix=True,
                        confusion_matrix_file=None):
    # Evaluate model based on hungarian matching between predicted cluster assignment and gt classes.
    # This is computed only for the passed subhead index.

    # Hungarian matching
    head = all_predictions[subhead_index]
    targets = head['targets'].cuda()
    predictions = head['predictions'].cuda()
    probs = head['probabilities'].cuda()
    num_classes = torch.unique(targets).numel()
    num_elems = targets.size(0)

    match = _hungarian_match(predictions, targets, preds_k=num_classes, targets_k=num_classes)
    reordered_preds = torch.zeros(num_elems, dtype=predictions.dtype).cuda()
    for pred_i, target_i in match:
        reordered_preds[predictions == int(pred_i)] = int(target_i)

    # Gather performance metrics
    acc = int((reordered_preds == targets).sum()) / float(num_elems)
    nmi = metrics.normalized_mutual_info_score(targets.cpu().numpy(), predictions.cpu().numpy())
    ari = metrics.adjusted_rand_score(targets.cpu().numpy(), predictions.cpu().numpy())
    
    _, preds_top5 = probs.topk(5, 1, largest=True)
    reordered_preds_top5 = torch.zeros_like(preds_top5)
    for pred_i, target_i in match:
        reordered_preds_top5[preds_top5 == int(pred_i)] = int(target_i)
    correct_top5_binary = reordered_preds_top5.eq(targets.view(-1,1).expand_as(reordered_preds_top5))
    top5 = float(correct_top5_binary.sum()) / float(num_elems)

    # Compute confusion matrix
    if compute_confusion_matrix:
        confusion_matrix(reordered_preds.cpu().numpy(), targets.cpu().numpy(), 
                            class_names, confusion_matrix_file)

    return {'ACC': acc, 'ARI': ari, 'NMI': nmi, 'ACC Top-5': top5, 'hungarian_match': match}


@torch.no_grad()
def _hungarian_match(flat_preds, flat_targets, preds_k, targets_k):
    # Based on implementation from IIC
    num_samples = flat_targets.shape[0]

    assert (preds_k == targets_k)  # one to one
    num_k = preds_k
    num_correct = np.zeros((num_k, num_k))

    for c1 in range(num_k):
        for c2 in range(num_k):
            # elementwise, so each sample contributes once
            votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
            num_correct[c1, c2] = votes

    # num_correct is small
    match = linear_sum_assignment(num_samples - num_correct)
    match = np.array(list(zip(*match)))

    # return as list of tuples, out_c to gt_c
    res = []
    for out_c, gt_c in match:
        res.append((out_c, gt_c))

    return res
