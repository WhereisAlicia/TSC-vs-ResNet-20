'''
Training script for Fashion-MNIST
'''
from __future__ import print_function

import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import shutil
import time
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.utils.data import random_split
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
# import torchvision.datasets as datasets
import models.fashion as models
import transforms
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Fashion-MNIST Training')
# Datasets
parser.add_argument('-d', '--dataset', default='fashionmnist', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150,225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet20)')
parser.add_argument('--depth', type=int, default=20, help='Model depth.')
parser.add_argument('--widen-factor', type=int, default=10, help='Widen factor. 10')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')


# Random Erasing
parser.add_argument('--p', default=0, type=float, help='Random Erasing probability')
parser.add_argument('--sh', default=0.4, type=float, help='max erasing area')
parser.add_argument('--r1', default=0.3, type=float, help='aspect of erasing area')


args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'fashionmnist'

# Use CUDA
use_cuda = torch.cuda.is_available()
if not use_cuda:
    print("No GPU detected. Please run this script on a machine with CUDA support.")
    exit()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_valid_acc = 0  # best accuracy

def main():
    patience = 35               
    no_improve_epochs = 0
    epoch_times = []
    global best_valid_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data
    # print('==> Preparing dataset %s' % args.dataset)
    # will not use
    transform_train = transforms.Compose([
        # transforms.RandomCrop(28, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,)),
        # transforms.RandomErasing(probability = args.p, sh = args.sh, r1 = args.r1, mean = [0.4914]),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,)),
    ])

    # if args.dataset == 'fashionmnist':
    #     dataloader = datasets.FashionMNIST
    #     num_classes = 10
    # trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
    # trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    # testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
    # testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    data_path = os.path.join('..','..', 'processed_data')

    X_train = np.load(os.path.join(data_path, 'X_train_2d_32.npy'))
    X_test = np.load(os.path.join(data_path, 'X_test_2d_32.npy'))
    Y_train = np.load(os.path.join(data_path, 'Y_train.npy'))
    Y_test = np.load(os.path.join(data_path, 'Y_test.npy'))

    print(f"Loaded preprocessed data: {X_train.shape}, {Y_train.shape}")

    # === Convert to torch tensors ===
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.long)
    Y_test = torch.tensor(Y_test, dtype=torch.long)

    # === Wrap into TensorDataset using the same style ===
    trainset_full = data.TensorDataset(X_train, Y_train)
    testset = data.TensorDataset(X_test, Y_test)

    valid_ratio = 0.1
    num_train = len(trainset_full)
    valid_size = int(np.floor(valid_ratio * num_train))
    train_size = num_train - valid_size

    train_dataset, valid_dataset = random_split(
        trainset_full, 
        [train_size, valid_size], 
        generator=torch.Generator().manual_seed(args.manualSeed)
    )

    print(f"Full Trainset size: {len(trainset_full)}")
    print(f"New Trainset size: {len(train_dataset)}")
    print(f"Validation set size: {len(valid_dataset)}")
    print(f"Testset size: {len(testset)}")

    # === DataLoaders ===
    trainloader = data.DataLoader(train_dataset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
    validloader = data.DataLoader(valid_dataset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers) # <-- 新增
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    num_classes = 10

    # Model   
    print("==> creating model '{}'".format(args.arch))
    if args.arch.startswith('wrn'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.endswith('resnet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                )

    model = torch.nn.DataParallel(model).cuda()



    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Resume
    title = 'fashionmnist-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        # GPU
        checkpoint = torch.load(args.resume)
        # CPU
        # checkpoint = torch.load(args.resume, map_location=None if use_cuda else 'cpu')

        best_valid_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Epoch','Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])


    if args.evaluate:
        print('\n Evaluation only')
        test_loss, test_acc,_,_ = test(testloader, model, criterion, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # Train and val
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
        train_start = time.time()
        train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, use_cuda)
        train_end = time.time()
        train_time = train_end - train_start

        valid_start = time.time()
        valid_loss, valid_acc, all_valid_targets, all_valid_preds = validate(validloader, model, criterion, epoch, use_cuda)
        valid_end = time.time()
        valid_time = valid_end - valid_start

        epoch_end = time.time()
        total_epoch_time = epoch_end - epoch_start

        epoch_times.append({
        'epoch': epoch + 1,
        'train_time_sec': train_time,
        'valid_time_sec': valid_time,
        'total_epoch_time_sec': total_epoch_time})

        # append logger file
        logger.append([epoch + 1, state['lr'], train_loss, valid_loss, train_acc, valid_acc])

        # save model
        is_best = valid_acc > best_valid_acc
        if is_best:
            best_valid_acc = valid_acc
            no_improve_epochs = 0
            print(f"*** New best validation accuracy: {valid_acc:.4f} at epoch {epoch + 1} ***")
        
            # cm = confusion_matrix(all_valid_targets, all_valid_preds)
            # confusion_matrix_data_path = os.path.join(args.checkpoint, 'confusion_matrix_quant_final_data.npy')
            # np.save(confusion_matrix_data_path, cm)
            # report = classification_report(all_valid_targets, all__valid_preds, digits=4, output_dict=True)
            # report_df = pd.DataFrame(report).T

            # excel_path = os.path.join(args.checkpoint, "best_classification_report.csv")
            # report_df.to_excel(excel_path, index=True)
            # print(f"Best classification report saved to {excel_path}")

            # class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
            #    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
            
            # plt.figure(figsize=(12, 10))
            # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
            # plt.title('Best Model Confusion Matrix')
            # plt.xlabel('Predicted Label')
            # plt.ylabel('True Label')
            # cm_pic_path = os.path.join(args.checkpoint, 'best_confusion_matrix.png')
            # plt.savefig(cm_pic_path)
            # plt.close()
            # # print(f"Best confusion matrix saved to {cm_pic_path}")
        else:
            no_improve_epochs += 1


        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': valid_acc,
                'best_acc': best_valid_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)
        
        # if no_improve_epochs >= patience:
        #     print(f"Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
        #     # print(f"No improvement for {patience} epochs, but continuing training (no early stop).")
        #     break
        


    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    # === Save training/testing time results ===
    df = pd.DataFrame(epoch_times)
    df.loc['Average'] = df.mean(numeric_only=True)
    df.loc['Total'] = df.sum(numeric_only=True)
    csv_path = os.path.join(args.checkpoint, 'epoch_train_valid_times.csv')
    df.to_csv(csv_path, index=True)
    print(f"Training and validation times saved to {csv_path}")

    print('Best Validation Acc:')
    print(best_valid_acc)

    print(f'==> Loading best model (Valid Acc: {best_valid_acc:.2f}%) for final test...')
    
    best_model_path = os.path.join(args.checkpoint, 'model_best.pth.tar')
    if not os.path.isfile(best_model_path):
        print("Error: 'model_best.pth.tar' not found. Cannot perform final test.")
        return
    
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['state_dict'])

    train_loss, train_acc, train_targets, train_preds = test(trainloader, model, criterion, use_cuda)
    train_report_dict = classification_report(train_targets, train_preds, target_names=class_names, digits=4, output_dict=True)
    train_report_df = pd.DataFrame(train_report_dict).transpose()
    train_report_df.to_csv(os.path.join(args.checkpoint, 'classification_report_cnn_train.csv'))
    
    # Train Confusion Matrix (Optional, just save pic)
    cm_train = confusion_matrix(train_targets, train_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix on Training Set (CNN)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(args.checkpoint, 'confusion_matrix_cnn_train.png'))
    plt.close()
    np.save(os.path.join(args.checkpoint, 'confusion_matrix_cnn_train.npy'), cm_train)


    # Valid Report
    valid_loss, valid_acc, valid_targets, valid_preds = test(validloader, model, criterion, use_cuda)
    valid_report_dict = classification_report(valid_targets, valid_preds, target_names=class_names, digits=4, output_dict=True)
    valid_report_df = pd.DataFrame(valid_report_dict).transpose()
    valid_report_df.to_csv(os.path.join(args.checkpoint, 'classification_report_cnn_valid.csv'))
  
    # test
    test_start_time = time.time()
    test_loss, test_acc, all_test_targets, all_test_preds = test(testloader, model, criterion, use_cuda)
    test_time = time.time() - test_start_time
    
    print('\n==> Final Test Results:')
    print(f'  Test Loss: {test_loss:.4f}')
    print(f'  Test Acc:  {test_acc:.2f}%')
    print(f'  Test Time: {test_time:.2f} sec')


    print('==> Saving final test reports...')
    report = classification_report(all_test_targets, all_test_preds, digits=4, output_dict=True)
    report_df = pd.DataFrame(report).T
    csv_path = os.path.join(args.checkpoint, "final_test_classification_report.csv")
    report_df.to_csv(csv_path, index=True)
    print(f"Final test classification report saved to {csv_path}")


    cm = confusion_matrix(all_test_targets, all_test_preds)
    

    confusion_matrix_data_path = os.path.join(args.checkpoint, 'final_test_confusion_matrix_data.npy')
    np.save(confusion_matrix_data_path, cm)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Final Test Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    cm_pic_path = os.path.join(args.checkpoint, 'final_test_confusion_matrix.png')
    plt.savefig(cm_pic_path)
    plt.close()
    print(f"Final test confusion matrix saved to {cm_pic_path} (and .npy data)")

    # --- 4. Error Analysis (Extract Misclassified) ---
    print("\n[Analysis] Extracting Misclassified Examples...")
    misclassified_mask = np.array(all_test_preds) != np.array(all_test_targets)
    misclassified_indices = np.where(misclassified_mask)[0]

    error_analysis_data = {
        'test_set_index': misclassified_indices,
        'true_label': np.array(all_test_targets)[misclassified_indices],
        'predicted_label': np.array(all_test_preds)[misclassified_indices],
        'true_class_name': [class_names[i] for i in np.array(all_test_targets)[misclassified_indices]],
        'predicted_class_name': [class_names[i] for i in np.array(all_test_preds)[misclassified_indices]]
    }
    error_df = pd.DataFrame(error_analysis_data)
    error_csv_path = os.path.join(args.checkpoint, 'error_analysis_cnn_misclassified.csv')
    error_df.to_csv(error_csv_path, index=False)
    print(f"Misclassified examples saved to {error_csv_path}")



    total_train_time = df.loc['Total', 'train_time_sec']
    total_valid_time = df.loc['Total', 'valid_time_sec']
    
    summary_path = os.path.join(args.checkpoint, 'summary_report.txt')
    with open(summary_path, 'w') as f:
        f.write("========== Summary Report ==========\n")
        f.write(f"Model Architecture: {args.arch} (Depth: {args.depth})\n")
        f.write(f"Checkpoint Path: {args.checkpoint}\n")
        f.write(f"Total Epochs Run: {args.epochs} (or stopped early)\n")
        f.write(f"Best Validation Epoch: {checkpoint['epoch']}\n")
        f.write(f"Best Validation Accuracy: {best_valid_acc:.4f}%\n")
        f.write("------------------------------------\n")
        f.write(f"Final Test Accuracy: {test_acc:.4f}%\n")
        f.write(f"Final Test Loss: {test_loss:.6f}\n")
        f.write("------------------------------------\n")
        f.write(f"Total Training Time: {total_train_time:.2f} sec\n")
        f.write(f"Total Validation Time: {total_valid_time:.2f} sec\n")
        f.write(f"Final Test Time: {test_time:.2f} sec\n")
    print(f"Summary report saved to {summary_path}")



def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)


        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def validate(valloader, model, criterion, epoch, use_cuda):

    start_time = time.time()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()

    all_preds = []
    all_targets = []

    bar = Bar('Processing', max=len(valloader))
    for batch_idx, (inputs, targets) in enumerate(valloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        # inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # compute output

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(valloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()

    total_time = time.time() - start_time
    print(f"validation time for epoch {epoch + 1}: {total_time:.2f} seconds")
    return (losses.avg, top1.avg,all_targets, all_preds)

def test(testloader, model, criterion, use_cuda):

    start_time = time.time()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()

    all_preds = []
    all_targets = []

    bar = Bar('Testing', max=len(testloader)) 
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()

    total_time = time.time() - start_time
    print(f"Final Testing time: {total_time:.2f} seconds") 
    return (losses.avg, top1.avg, all_targets, all_preds)


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()
