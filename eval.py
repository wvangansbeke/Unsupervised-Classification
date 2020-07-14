"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import argparse
import torch
import yaml
from termcolor import colored
from utils.common_config import get_val_dataset, get_val_transformations, get_val_dataloader,\
                                get_model
from utils.evaluate_utils import get_predictions, hungarian_evaluate
from utils.memory import MemoryBank 
from utils.utils import fill_memory_bank
from PIL import Image

FLAGS = argparse.ArgumentParser(description='Evaluate models from the model zoo')
FLAGS.add_argument('--config_exp', help='Location of config file')
FLAGS.add_argument('--model', help='Location where model is saved')
FLAGS.add_argument('--visualize_prototypes', action='store_true', 
                    help='Show the prototpye for each cluster')
args = FLAGS.parse_args()

def main():
    
    # Read config file
    print(colored('Read config file {} ...'.format(args.config_exp), 'blue'))
    with open(args.config_exp, 'r') as stream:
        config = yaml.safe_load(stream)
    config['batch_size'] = 512 # To make sure we can evaluate on a single 1080ti
    print(config)

    # Get dataset
    print(colored('Get validation dataset ...', 'blue'))
    transforms = get_val_transformations(config)
    dataset = get_val_dataset(config, transforms)
    dataloader = get_val_dataloader(config, dataset)
    print('Number of samples: {}'.format(len(dataset)))

    # Get model
    print(colored('Get model ...', 'blue'))
    model = get_model(config)
    print(model)

    # Read model weights
    print(colored('Load model weights ...', 'blue'))
    state_dict = torch.load(args.model, map_location='cpu')

    if config['setup'] in ['simclr', 'moco', 'selflabel']:
        model.load_state_dict(state_dict)

    elif config['setup'] == 'scan':
        model.load_state_dict(state_dict['model'])

    else:
        raise NotImplementedError
        
    # CUDA
    model.cuda()

    # Perform evaluation
    if config['setup'] in ['simclr', 'moco']:
        print(colored('Perform evaluation of the pretext task (setup={}).'.format(config['setup']), 'blue'))
        print('Create Memory Bank')
        if config['setup'] == 'simclr': # Mine neighbors after MLP
            memory_bank = MemoryBank(len(dataset), config['model_kwargs']['features_dim'],
                                    config['num_classes'], config['criterion_kwargs']['temperature'])

        else: # Mine neighbors before MLP
            memory_bank = MemoryBank(len(dataset), config['model_kwargs']['features_dim'], 
                                    config['num_classes'], config['temperature'])
        memory_bank.cuda()

        print('Fill Memory Bank')
        fill_memory_bank(dataloader, model, memory_bank)

        print('Mine the nearest neighbors')
        for topk in [1, 5, 20]: # Similar to Fig 2 in paper 
            _, acc = memory_bank.mine_nearest_neighbors(topk)
            print('Accuracy of top-{} nearest neighbors on validation set is {:.2f}'.format(topk, 100*acc))


    elif config['setup'] in ['scan', 'selflabel']:
        print(colored('Perform evaluation of the clustering model (setup={}).'.format(config['setup']), 'blue'))
        head = state_dict['head'] if config['setup'] == 'scan' else 0
        predictions, features = get_predictions(config, dataloader, model, return_features=True)
        clustering_stats = hungarian_evaluate(head, predictions, dataset.classes, 
                                                compute_confusion_matrix=True)
        print(clustering_stats)
        if args.visualize_prototypes:
            prototype_indices = get_prototypes(config, predictions[head], features, model)
            visualize_indices(prototype_indices, dataset, clustering_stats['hungarian_match'])
    else:
        raise NotImplementedError

@torch.no_grad()
def get_prototypes(config, predictions, features, model, topk=10):
    import torch.nn.functional as F

    # Get topk most certain indices and pred labels
    print('Get topk')
    probs = predictions['probabilities']
    n_classes = probs.shape[1]
    dims = features.shape[1]
    max_probs, pred_labels = torch.max(probs, dim = 1)
    indices = torch.zeros((n_classes, topk))
    for pred_id in range(n_classes):
        probs_copy = max_probs.clone()
        mask_out = ~(pred_labels == pred_id)
        probs_copy[mask_out] = -1
        conf_vals, conf_idx = torch.topk(probs_copy, k = topk, largest = True, sorted = True)
        indices[pred_id, :] = conf_idx

    # Get corresponding features
    selected_features = torch.index_select(features, dim=0, index=indices.view(-1).long())
    selected_features = selected_features.unsqueeze(1).view(n_classes, -1, dims)

    # Get mean feature per class
    mean_features = torch.mean(selected_features, dim=1)

    # Get min distance wrt to mean
    diff_features = selected_features - mean_features.unsqueeze(1)
    diff_norm = torch.norm(diff_features, 2, dim=2)

    # Get final indices
    _, best_indices = torch.min(diff_norm, dim=1)
    one_hot = F.one_hot(best_indices.long(), indices.size(1)).byte()
    proto_indices = torch.masked_select(indices.view(-1), one_hot.view(-1))
    proto_indices = proto_indices.int().tolist()
    return proto_indices

def visualize_indices(indices, dataset, hungarian_match):
    import matplotlib.pyplot as plt
    import numpy as np

    for idx in indices:
        img = np.array(dataset.get_image(idx)).astype(np.uint8)
        img = Image.fromarray(img)
        plt.figure()
        plt.axis('off')
        plt.imshow(img)
        plt.show()


if __name__ == "__main__":
    main() 
