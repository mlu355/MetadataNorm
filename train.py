import argparse
import glob 
import os
import sys

import dcor
import numpy as np
import numpy.random as random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms, utils

from metadatanorm import MetadataNorm
from synthetic_dataset import SyntheticDataset
from models import *
from utils import *

parser = argparse.ArgumentParser(description='Metadata Normalization demonstrated on Synthetic Dataset')
parser.add_argument('--mdn', type=str, default='Conv', help='Type of MDN config [Baseline, Linear, Conv]')
parser.add_argument('--model_dir', type=str, default='experiments/default', help='directory name to store output')
parser.add_argument('--batch_size', type=int, default=200, help='batch size')
parser.add_argument('--epochs', type=int, default=500, help='number of epochs')
parser.add_argument('--N', type=int, default=1000, help='size of each group (group A or group B)')
parser.add_argument('--runs', type=int, default=1, help='number of experimental runs')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
args = parser.parse_args()

# Set up and run experiment for a single model configuration
def run_experiment(mdn, 
                   run_name_base, 
                   batch_size, 
                   learning_rate, 
                   run, 
                   x, 
                   labels, 
                   cf,
                   x_val, 
                   labels_val, 
                   cf_val,
                   epochs=5000, 
                   N=1000):
        
    trainset_size = 2 * N
    experiment_name = os.path.join(run_name_base, 'batch_size' + str(batch_size))
    run_name = os.path.join(experiment_name, 'run' + str(run))
    log_file = os.path.join(experiment_name, 'metrics.txt')
    skmetrics_file = os.path.join(experiment_name, 'skmetrics.txt')
    run_log_file = os.path.join(run_name, 'metrics.txt')
    if not os.path.exists(run_name):
        os.makedirs(run_name)
    with open(run_log_file, 'w') as f:
        f.write('acc' + "\t" + 'dc0_val' + '\t' + 'dc1_val' + '\t' + 'loss' + "\n")
    print("run name:", run_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    
    # Calculate confounder kernel, the precalculated kernel (X^TX)^-1 for MDN based on the vector X 
    # of confounders. Only needs to be calculated once before training. 
    if mdn == 'Conv' or mdn == 'Linear':
        X = np.zeros((N*2,3))
        X[:,0] = labels
        X[:,1] = cf
        X[:,2] = np.ones((N*2,))
        XTX = np.transpose(X).dot(X)
        kernel = np.linalg.inv(XTX)
        cf_kernel = nn.Parameter(torch.tensor(kernel).float().to(device), requires_grad=False)

    # Create model
    if mdn == 'Baseline':
        model = BaselineNet()
    elif mdn == 'Linear':
        model = MDN_Linear(2*N, batch_size, cf_kernel)
    elif mdn == 'Conv':
        model = MDN_Conv(2*N, batch_size, cf_kernel)
    else:
        print('mdn type not supported')
        return 
    
    model.to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr=1e-6, factor=0.5)
    iterations = 2 * N // batch_size
    print(model)
    
    # Make dataloaders
    print("Making dataloaders...")
    train_set = SyntheticDataset(x, labels, cf)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size, shuffle=True, pin_memory=True)
    val_set = SyntheticDataset(x_val, labels_val, cf_val)
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size, shuffle=True, pin_memory=True)

    # Run training 
    acc_history = []
    acc_history_val = []
    dc0s_val = []
    dc1s_val = []
    dcors = []
    losses = []
    losses_val = []
    patience = 0 # number of epochs where val loss doesn't decrease
    min_loss = float('inf')
    run_dir = os.path.join('plot_data', run_name)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
        
    for e in range(epochs):
        cfs_val = []
        feature_val = []
        epoch_acc = 0
        epoch_acc_val = 0
        epoch_loss = 0
        epoch_loss_val = 0
        pred_vals = []
        target_vals = []

        # Training pass
        model = model.train()
        for i, sample_batched in enumerate(train_loader):
            data = sample_batched['image'].float()
            target = sample_batched['label'].float()
            cf_batch = sample_batched['cfs'].float()
            data, target = data.cuda(), target.cuda()

            # Add confounder input feature (cfs) to model. cfs are stored in the dataset and need be set 
            # during training for each batch.
            X_batch = np.zeros((batch_size,3))
            X_batch[:,0] = target.cpu().detach().numpy()
            X_batch[:,1] = cf_batch.cpu().detach().numpy()
            X_batch[:,2] = np.ones((batch_size,))
            with torch.no_grad():
                model.cfs = nn.Parameter(torch.Tensor(X_batch).to(device), requires_grad=False)

            # Forward pass
            optimizer.zero_grad()
            y_pred, fc = model(data)
            loss = criterion(y_pred, target.unsqueeze(1))
            acc = binary_acc(y_pred, target.unsqueeze(1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_acc += acc.item()

        # Validation pass
        model = model.eval()
        for i, sample_batched in enumerate(val_loader):
            data = sample_batched['image'].float()
            target = sample_batched['label'].float()
            cf_batch = sample_batched['cfs'].float()
            data, target = data.cuda(), target.cuda()

            X_batch = np.zeros((batch_size,3))
            X_batch[:,0] = target.cpu().detach().numpy()
            X_batch[:,1] = cf_batch.cpu().detach().numpy()
            X_batch[:,2] = np.ones((batch_size,))

            with torch.no_grad():
                model.cfs = nn.Parameter(torch.Tensor(X_batch).to(device), requires_grad=False)
                y_pred, fc = model(data)
                loss = criterion(y_pred, target.unsqueeze(1))
                acc = binary_acc(y_pred, target.unsqueeze(1))
                epoch_loss_val += loss.item()
                epoch_acc_val += acc.item()

            # Save learned features
            feature_val.append(fc)
            cfs_val.append(cf_batch)
            target_vals.append(target.cpu())
            pred_vals.append(y_pred.cpu())
            
        # Calculate distance correlation between confounders and learned features
        epoch_targets = np.concatenate(target_vals, axis=0)
        epoch_preds = np.concatenate(pred_vals, axis=0)
        i0_val = np.where(epoch_targets == 0)[0]
        i1_val = np.where(epoch_targets == 1)[0]
        epoch_layer = np.concatenate(feature_val, axis=0)
        epoch_cf = np.concatenate(cfs_val, axis=0)
        dc0_val = dcor.u_distance_correlation_sqr(epoch_layer[i0_val], epoch_cf[i0_val])
        dc1_val = dcor.u_distance_correlation_sqr(epoch_layer[i1_val], epoch_cf[i1_val])
        print("correlations for feature 0:", dc0_val)
        print("correlations for feature 1:", dc1_val)
        dc0s_val.append(dc0_val)
        dc1s_val.append(dc1_val)

        curr_acc = epoch_acc/iterations
        acc_history.append(curr_acc)
        losses.append(epoch_loss)
        curr_acc_val = epoch_acc_val/iterations
        acc_history_val.append(curr_acc_val)
        losses_val.append(epoch_loss_val)

        print('learning rate:', optimizer.param_groups[0]['lr'])
        lr_scheduler.step(epoch_loss_val)


        # Save model with lowest loss
        if epoch_loss_val + 0.001 < min_loss:
            print("Best loss so far! Saving model...")
            min_loss = epoch_loss_val
            #if run <= 5:
                #torch.save(model, os.path.join('plot_data', run_name, 'best_model.pth'))
            patience = 0
            
        print(f'Train: Epoch {e+0:03}: | Loss: {epoch_loss/iterations:.5f} | Acc: {epoch_acc/iterations:.3f}')
        print(f'Val: Epoch {e+0:03}: | Loss: {epoch_loss_val/iterations:.5f} | Acc: {epoch_acc_val/iterations:.3f}')
        np.save(os.path.join(run_name, 'd' + str(trainset_size) + '.npy'), acc_history)
        np.save(os.path.join(run_name, 'loss_d' + str(trainset_size)) + '.npy', losses)
        np.save(os.path.join(run_name, 'val_loss_d' + str(trainset_size)) + '.npy', losses_val)
        np.save(os.path.join(run_name, 'val_acc_d' + str(trainset_size) + '.npy'), acc_history_val)
        np.save(os.path.join(run_name, 'val_dc0s_d' + str(trainset_size) + '.npy'), dc0s_val)
        np.save(os.path.join(run_name, 'val_dc1s_d' + str(trainset_size) + '.npy'), dc1s_val)
        
        with open(run_log_file, 'a') as f:
            f.write(str(curr_acc_val) + '\t' + str(dc0_val) + '\t' + str(dc1_val) + '\t' + str(epoch_loss_val) + '\n')
            
        print("patience:", patience)
        patience += 1
        if patience > 200:
            print("out of patience")
            break
            
    y_test, y_pred_list = test(model, mdn, batch_size, run=run, N=N)
    
    with open(log_file, 'a') as f:
        f.write(str(run) + '\t' + str(curr_acc_val) + '\t' + str(dc0_val) + '\t' + str(dc1_val) + '\t' + str(epoch_loss_val) + "\n")
    
    with open(skmetrics_file, 'a') as f:
        f.write("Report for run " + str(run))
        f.write(classification_report(y_test, y_pred_list, digits=3) + '\n')

# Test model
def test(model, mdn, batch_size, run=None, N=2000):
    
    # Get model
    if isinstance(model, str):
        run_name_base = model
        run_name = os.path.join(run_name_base, 'batch_size' + str(batch_size), 'run' + str(run))
        model = torch.load(os.path.join('plot_data', run_name, 'best_model.pth'))
        
    # Generate test set
    y_test, cf_test, cf2_test, mf_test, x_test, y_test = generate_data(N, seed=args.seed+3)
    X_test = torch.from_numpy(np.swapaxes(x_test, 1, 3)).float()
    test_set = SyntheticDataset(X_test, y_test, cf_test)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size, shuffle=False,pin_memory=True)
    
    # Run evaluation
    y_pred_list = [] 
    model.eval() 
    model.to('cpu')
    with torch.no_grad():
        for i, sample_batched in enumerate(test_loader):
            data = sample_batched['image'].float()
            target = sample_batched['label'].float()
            cf_batch = sample_batched['cfs'].float()

            if mdn:
                X_batch = np.zeros((batch_size,3))
                X_batch[:,0] = target.cpu().detach().numpy()
                X_batch[:,1] = cf_batch.cpu().detach().numpy()
                X_batch[:,2] = np.ones((batch_size,))
                model.cfs = nn.Parameter(torch.Tensor(X_batch), requires_grad=False)

            y_test_pred, _ = model(data)
            y_pred_tag = torch.round(y_test_pred)
            y_pred_list.append(y_pred_tag.cpu())

    # Create evaluation
    y_pred_list = np.array([a.squeeze().tolist() for a in y_pred_list]).flatten()
    print(classification_report(y_test, y_pred_list, digits=3))
    print(confusion_matrix(y_test, y_pred_list)) 
    return y_test, y_pred_list


def run_experiments():
    
    # Initialize values
    batch_size = args.batch_size
    mdn = args.mdn
    runs = args.runs
    learning_rate = args.lr
    epochs = args.epochs
    N = args.N
    np.random.seed(args.seed)

    # Generate training and validation data
    labels, cf, _, _, x, y = generate_data(N, seed=args.seed)
    labels_val, cf_val, _, _, x_val, y_val = generate_data(N, seed=args.seed+1)
    x = np.swapaxes(x, 1, 3) # move channels after batch so we have (N, channels, h, w)
    x_val = np.swapaxes(x_val, 1, 3)

    # Run experiments
    run_name_base = os.path.join(args.model_dir, mdn)
    experiment_name = os.path.join(run_name_base, 'batch_size' + str(batch_size))
    log_file = os.path.join(experiment_name, 'metrics.txt')
    skmetrics_file = os.path.join(experiment_name, 'skmetrics.txt')
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)
    print(experiment_name)

    with open(log_file, 'a') as f:
        f.write('mdn=' + mdn + "\t" 
                + 'batch_size=' + str(batch_size) + "\t" 
                + "lr=" + str(learning_rate) + "\t"
                + "N="  + str(N) + "\n")
        f.write('run' + '\t' + 'acc' + "\t" + 'dc0_val' + '\t' + 'dc1_val' + '\t' + 'loss' + "\n")

    with open(skmetrics_file, 'a') as f:
        f.write("sklearn metrics\n")

    for i in range(1, runs+1):
        print("\nRunning experiment " + mdn + ": Run " + str(i))
        print('-----------------------------------------------------------')
        run_experiment(mdn=mdn, 
                       run_name_base=run_name_base, 
                       batch_size=batch_size, 
                       learning_rate=learning_rate, 
                       run=i, 
                       epochs=epochs, 
                       x=x,
                       labels=labels,
                       cf=cf,
                       x_val=x_val,
                       labels_val=labels_val,
                       cf_val=cf_val,
                       N=N)

if __name__ == "__main__":
    run_experiments()
