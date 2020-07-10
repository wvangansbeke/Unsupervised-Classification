"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import argparse
import os
import torch

from termcolor import colored
from utils.config import create_config
from utils.common_config import get_train_transformations, get_val_transformations,\
                                get_train_dataset, get_train_dataloader,\
                                get_val_dataset, get_val_dataloader,\
                                get_optimizer, get_model, get_criterion,\
                                adjust_learning_rate
from utils.evaluate_utils import get_predictions, scan_evaluate, hungarian_evaluate
from utils.train_utils import scan_train

FLAGS = argparse.ArgumentParser(description='SCAN Loss')
FLAGS.add_argument('--config_env', help='Location of path config file')
FLAGS.add_argument('--config_exp', help='Location of experiments config file')

def main():
    args = FLAGS.parse_args()
    p = create_config(args.config_env, args.config_exp)
    print(colored(p, 'red'))

    # CUDNN
    torch.backends.cudnn.benchmark = True

    # Data
    print(colored('Get dataset and dataloaders', 'blue'))
    train_transformations = get_train_transformations(p)
    val_transformations = get_val_transformations(p)
    train_dataset = get_train_dataset(p, train_transformations, 
                                        split='train', to_neighbors_dataset = True)
    val_dataset = get_val_dataset(p, val_transformations, to_neighbors_dataset = True)
    train_dataloader = get_train_dataloader(p, train_dataset)
    val_dataloader = get_val_dataloader(p, val_dataset)
    print('Train transforms:', train_transformations)
    print('Validation transforms:', val_transformations)
    print('Train samples %d - Val samples %d' %(len(train_dataset), len(val_dataset)))
    
    # Model
    print(colored('Get model', 'blue'))
    model = get_model(p, p['pretext_model'])
    print(model)
    model = torch.nn.DataParallel(model)
    model = model.cuda()

    # Optimizer
    print(colored('Get optimizer', 'blue'))
    optimizer = get_optimizer(p, model, p['update_cluster_head_only'])
    print(optimizer)
    
    # Warning
    if p['update_cluster_head_only']:
        print(colored('WARNING: SCAN will only update the cluster head', 'red'))

    # Loss function
    print(colored('Get loss', 'blue'))
    criterion = get_criterion(p) 
    criterion.cuda()
    print(criterion)

    # Checkpoint
    if os.path.exists(p['scan_checkpoint']):
        print(colored('Restart from checkpoint {}'.format(p['scan_checkpoint']), 'blue'))
        checkpoint = torch.load(p['scan_checkpoint'], map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])        
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        best_loss_head = checkpoint['best_loss_head']

    else:
        print(colored('No checkpoint file at {}'.format(p['scan_checkpoint']), 'blue'))
        start_epoch = 0
        best_loss = 1e4
        best_loss_head = None
 
    # Main loop
    print(colored('Starting main loop', 'blue'))

    for epoch in range(start_epoch, p['epochs']):
        print(colored('Epoch %d/%d' %(epoch+1, p['epochs']), 'yellow'))
        print(colored('-'*15, 'yellow'))

        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))

        # Train
        print('Train ...')
        scan_train(train_dataloader, model, criterion, optimizer, epoch, p['update_cluster_head_only'])

        # Evaluate 
        print('Make prediction on validation set ...')
        predictions = get_predictions(p, val_dataloader, model)

        print('Evaluate based on SCAN loss ...')
        scan_stats = scan_evaluate(predictions)
        print(scan_stats)
        lowest_loss_head = scan_stats['lowest_loss_head']
        lowest_loss = scan_stats['lowest_loss']
       
        if lowest_loss < best_loss:
            print('New lowest loss on validation set: %.4f -> %.4f' %(best_loss, lowest_loss))
            print('Lowest loss head is %d' %(lowest_loss_head))
            best_loss = lowest_loss
            best_loss_head = lowest_loss_head
            torch.save({'model': model.module.state_dict(), 'head': best_loss_head}, p['scan_model'])

        else:
            print('No new lowest loss on validation set: %.4f -> %.4f' %(best_loss, lowest_loss))
            print('Lowest loss head is %d' %(best_loss_head))

        print('Evaluate with hungarian matching algorithm ...')
        clustering_stats = hungarian_evaluate(lowest_loss_head, predictions, compute_confusion_matrix=False)
        print(clustering_stats)     

        # Checkpoint
        print('Checkpoint ...')
        torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(), 
                    'epoch': epoch + 1, 'best_loss': best_loss, 'best_loss_head': best_loss_head},
                     p['scan_checkpoint'])
    
    # Evaluate and save the final model
    print(colored('Evaluate best model based on SCAN metric at the end', 'blue'))
    model_checkpoint = torch.load(p['scan_model'], map_location='cpu')
    model.module.load_state_dict(model_checkpoint['model'])
    predictions = get_predictions(p, val_dataloader, model)
    clustering_stats = hungarian_evaluate(model_checkpoint['head'], predictions, 
                            class_names=val_dataset.dataset.classes, 
                            compute_confusion_matrix=True, 
                            confusion_matrix_file=os.path.join(p['scan_dir'], 'confusion_matrix.png'))
    print(clustering_stats)         
    
if __name__ == "__main__":
    main()
