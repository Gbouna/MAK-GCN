from __future__ import print_function
import os
import argparse
import torch
import random
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from importlib import import_module
import numpy as np
import sklearn.metrics as metrics
from util import cal_loss, IOStream
from data import MMRActionData
from ptflops import get_model_complexity_info

def init_seed(seed):
    """
    Set random seeds for reproducibility across torch, numpy, random.
    """
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

TRAIN_NAME = os.path.splitext(os.path.basename(__file__))[0]
def parse_arguments():
    parser = argparse.ArgumentParser(description='Action Recognition Using Sparse Point Cloud')
    parser.add_argument('--name', type=str, default='', help='Name of the experiment')
    parser.add_argument('--model', type=str, default='MAKGCN', help='Model architecture to use')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--test_batch_size', type=int, default=16, help='Test batch size')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--Tmax', type=int, default=250, help='Maximum scheduler iteration (for CosineAnnealingLR).')
    parser.add_argument('--use_sgd', action='store_true', help='Use SGD optimizer instead of Adam')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducibility')
    parser.add_argument('--eval', action='store_true', help='Run evaluation')
    parser.add_argument('--num_points', type=int, default=1100, help='Number of points per sample (max_points * stacks).')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, help='Dimension of latent embeddings')
    parser.add_argument('--k', type=int, default=20, help='Number of nearest neighbors (k) for graph construction')
    parser.add_argument('--model_path', type=str, default='', help='Path to a pretrained model')
    parser.add_argument('--gpu_idx', type=int, default=0, help='GPU index (set to -1 for CPU, 0 for first GPU, etc.)')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of multi-head kernel filters (for MAK layer)')
    return parser.parse_args()

def _init_(args):
    """
    Initialize the directory for storing models and backups.
    """
    if args.name == '':
        args.name = TRAIN_NAME
    model_dir = os.path.join('models', args.name)
    os.makedirs(os.path.join(model_dir, 'models'), exist_ok=True)
    os.system(f'cp {TRAIN_NAME}.py {model_dir}/{TRAIN_NAME}.py.backup')
    os.system(f'cp {args.model}.py {model_dir}/{args.model}.py.backup')
    os.system(f'cp util.py {model_dir}/util.py.backup')
    os.system(f'cp data.py {model_dir}/data.py.backup')

def train(args, io):
    """
    Train the model using the specified arguments and log with IOStream.
    """
    device = torch.device('cpu' if args.gpu_idx < 0 else f'cuda:{args.gpu_idx}')
    MODEL = import_module(args.model)

    if args.gpu_idx >= 0:
        io.cprint(f'Using GPU: {args.gpu_idx}')
    else:
        io.cprint('Using CPU')

    # Configure dataset
    root_dir = ''  # Current directory or specify directory
    mmr_dataset_config = {
        'processed_data': 'data/processed/mmr_action_head_2/data.pkl',
        'stacks': 50,  
        'max_points': 22,
        'zero_padding': 'per_data_point',
        'seed': 42,
        'forced_rewrite': True
    }

    # Load training data
    train_loader = DataLoader(
        MMRActionData(root=root_dir, partition='train', mmr_dataset_config=mmr_dataset_config),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True
    )

    # Load validation data
    val_loader = DataLoader(
        MMRActionData(root=root_dir, partition='val', mmr_dataset_config=mmr_dataset_config),
        batch_size=args.test_batch_size,
        shuffle=False,
        drop_last=False
    )

    # Initialize the model
    io.cprint(f'Using model: {args.model}')
    model = MODEL.Net(args)
    model = model.to(device)
    print(model)

    # Choose optimizer
    if args.use_sgd:
        io.cprint("Using SGD optimizer")
        optimizer = optim.SGD(model.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
    else:
        io.cprint("Using Adam optimizer")
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, args.Tmax, eta_min=args.lr)

    criterion = cal_loss
    best_val_loss = float('inf')
    patience_counter = 0 

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        count = 0.0
        train_pred = []
        train_true = []

        for data, label in train_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)  
            batch_size = data.size(0)
            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        train_acc = metrics.accuracy_score(train_true, train_pred)
        avg_train_acc = metrics.balanced_accuracy_score(train_true, train_pred)
        learning_rate = optimizer.param_groups[0]['lr']
        io.cprint(f'EPOCH {epoch}  lr = {learning_rate:.6f}')
        io.cprint(f'Train {epoch}, loss: {train_loss / count:.6f}, '
                  f'train acc: {train_acc:.6f}, train avg acc: {avg_train_acc:.6f}')
        # Validation
        model.eval()
        val_loss = 0.0
        count = 0.0
        val_pred = []
        val_true = []

        with torch.no_grad():
            for data, label in val_loader:
                data, label = data.to(device), label.to(device).squeeze()
                data = data.permute(0, 2, 1)
                batch_size = data.size(0)
                logits = model(data)
                loss = criterion(logits, label)
                preds = logits.max(dim=1)[1]
                count += batch_size
                val_loss += loss.item() * batch_size
                val_true.append(label.cpu().numpy())
                val_pred.append(preds.detach().cpu().numpy())

        val_true = np.concatenate(val_true)
        val_pred = np.concatenate(val_pred)
        val_acc = metrics.accuracy_score(val_true, val_pred)
        avg_val_acc = metrics.balanced_accuracy_score(val_true, val_pred)
        io.cprint(f'Val {epoch}, loss: {val_loss / count:.6f}, '
                  f'val acc: {val_acc:.6f}, val avg acc: {avg_val_acc:.6f}')
        
        # Early stopping check
        current_val_loss = val_loss / count
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), f'models/{args.name}/models/model.t7')
            io.cprint(f'Best model saved to models/{args.name}/models/model.t7')
        else:
            patience_counter += 1
            io.cprint(f'EarlyStopping counter: {patience_counter} out of {args.patience}')
            if patience_counter >= args.patience:
                io.cprint('Early stopping triggered')
                break  
         # Scheduler update
        if epoch < args.Tmax:
            scheduler.step()
        elif epoch == args.Tmax:
            # Switch to smaller learning rate
            for group in optimizer.param_groups:
                group['lr'] = 0.0001

def test(args, io):
    """
    Test the model using the best saved weights, and compute FLOPs.
    """
    MODEL = import_module(args.model)
    device = torch.device('cpu' if args.gpu_idx < 0 else f'cuda:{args.gpu_idx}')

    # Dataset configuration
    root_dir = ''  # Current directory or specify directory
    mmr_dataset_config = {
        'processed_data': 'data/processed/mmr_action_head_2/data.pkl',
        'stacks': 50,
        'max_points': 22,
        'zero_padding': 'per_data_point',
        'seed': 42,
        'forced_rewrite': True
    }

    # Load test data
    test_loader = DataLoader(
        MMRActionData(root=root_dir, partition='test', mmr_dataset_config=mmr_dataset_config),
        batch_size=args.test_batch_size,
        shuffle=False,
        drop_last=False
    )

    io.cprint('********** TEST STAGE **********')
    io.cprint('Loading the best model for evaluation')

    # Load the best model
    model = MODEL.Net(args)
    model = model.to(device)

    model.load_state_dict(torch.load(f'models/{args.name}/models/model.t7'))
    model.eval()

     # Compute FLOPs
    net_for_flops = model.module if hasattr(model, 'module') else model
    input_shape = (3, args.num_points) # (channels, num_points)

    macs, params = get_model_complexity_info(
        net_for_flops, 
        input_res=input_shape, 
        as_strings=True,             
        print_per_layer_stat=False, 
        verbose=False
    )
    io.cprint(f'[FLOPs Computation] MACs: {macs}, Params: {params}')
    # Evaluate on test set
    test_pred = []
    test_true = []

    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            logits = model(data)
            preds = logits.max(dim=1)[1]
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_test_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    test_precision, test_recall, test_f1, _ = metrics.precision_recall_fscore_support(
        test_true, test_pred, average='macro')
    io.cprint(f'Test Accuracy: {test_acc:.6f}, Test Avg Accuracy: {avg_test_acc:.6f}')
    io.cprint(f'Test Precision: {test_precision:.6f}, Test Recall: {test_recall:.6f}, Test F1: {test_f1:.6f}')


if __name__ == "__main__":
    args = parse_arguments()
    init_seed(args.seed)
    _init_(args)
    io = IOStream(f'models/{args.name}/train.log')
    io.cprint(str(args))

    if not args.eval:
        train(args, io)
    else:
        test(args, io)