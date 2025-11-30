import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from torchtime.models import InceptionTime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

use_cuda = torch.cuda.is_available()
if not use_cuda:
    print("No GPU detected. Please run this script on a machine with CUDA support.")
    exit()


GLOBAL_SEED = 2025
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
torch.cuda.manual_seed_all(GLOBAL_SEED)
data_path = os.path.join('..', 'processed_data')
output_dir = os.path.join(data_path, 'output/inceptiontime')  
os.makedirs(output_dir, exist_ok=True)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
n_classes = len(class_names)

def eval_loader(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)  # (B, n_classes)
            loss = criterion(logits, yb)
            total_loss += loss.item() * xb.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc

# Load best & Test evaluation
# Load best model (by val_loss)
def ensemble_evaluate(loader, n_models, device, desc="Predicting"):
    print(f"{desc}...")
    models = []
    for i in range(n_models):
        model = InceptionTime(
                n_inputs=n_inputs, 
                n_classes=n_classes,
                use_residual=True, 
                use_bottleneck=True, 
                depth=6, 
                n_convolutions = 3, 
                n_filters= 32, 
                kernel_size=32).to(device)
        model.load_state_dict(torch.load(os.path.join(output_dir, f'best_inceptiontime_{i+1}.pth'), map_location=device))
        model.eval()
        models.append(model)

    all_preds = []
    all_true  = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            
            probs_each = []
            for m in models:
                logits = m(xb)
                probs = torch.softmax(logits, dim=1)
                probs_each.append(probs)
                

            probs_each = torch.stack(probs_each, dim=0) # (5, batch_size)
            avg_probs = probs_each.mean(dim=0)
            final_preds = torch.argmax(avg_probs, dim=1)
            all_preds.append(final_preds.cpu().numpy())
            all_true.append(yb.numpy())

    return np.concatenate(all_true, axis=0), np.concatenate(all_preds, axis=0)

# Load data
X_train_full = np.load(os.path.join(data_path, 'X_train_1d.npy')).astype(np.float32)  # (60000, 1, 1024)
X_test = np.load(os.path.join(data_path, 'X_test_1d.npy')).astype(np.float32)         # (10000, 1, 1024)
Y_train_full = np.load(os.path.join(data_path, 'Y_train.npy')).astype(np.int64)       # (60000,)
Y_test = np.load(os.path.join(data_path, 'Y_test.npy')).astype(np.int64)              # (10000,)

# print(f"X_train_full shape: {X_train_full.shape}, dtype: {X_train_full.dtype}")
# print(f"X_test        shape: {X_test.shape},        dtype: {X_test.dtype}")
# print(f"Y_train_full  shape: {Y_train_full.shape},  dtype: {Y_train_full.dtype}")
# print(f"Y_test        shape: {Y_test.shape},        dtype: {Y_test.dtype}")


# Split train/valid (0.1 from TRAIN, stratified)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=GLOBAL_SEED)
(train_idx, valid_idx), = sss.split(X_train_full, Y_train_full)

X_train = X_train_full[train_idx]
Y_train = Y_train_full[train_idx]
X_valid = X_train_full[valid_idx]
Y_valid = Y_train_full[valid_idx]

print(f"Train: {X_train.shape}, Valid: {X_valid.shape}, Test: {X_test.shape}")


# Convert to tensors
X_train_t = torch.tensor(X_train)        
Y_train_t = torch.tensor(Y_train)        
X_valid_t = torch.tensor(X_valid)
Y_valid_t = torch.tensor(Y_valid)
X_test_t  = torch.tensor(X_test)
Y_test_t  = torch.tensor(Y_test)

train_ds = TensorDataset(X_train_t, Y_train_t)
valid_ds = TensorDataset(X_valid_t, Y_valid_t)
test_ds  = TensorDataset(X_test_t,  Y_test_t)

BATCH_SIZE = 256
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=8, pin_memory=use_cuda)
valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=use_cuda)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=use_cuda)

history = {
    'model':[],
    'epoch': [],
    'train_loss': [],
    'valid_loss': [],
    'epoch_train_time': [],
    'epoch_valid_time': [],
    'train_acc': [],
    'valid_acc': [],
    'lr': []
}

final_train_time=0
final_infer_time = 0
N_EPOCHS = 100
n_inputs = X_train.shape[1]
n_models=5
patience = 35
for i in range(n_models):
    model = InceptionTime(
        n_inputs=n_inputs, 
        n_classes=n_classes,
        use_residual=True, 
        use_bottleneck=True, 
        depth=6, 
        n_convolutions = 3, 
        n_filters= 32, 
        kernel_size=32).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=15,
        min_lr=1e-5,
        verbose=True)


    # Training loop with EarlyStopping (monitor val_loss) 

    best_val_acc = 0
    best_epoch = -1
    epochs_no_improve = 0

    print("Starting training...")
    start_time_train = time.time()

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        start_time_train_epoch = time.time()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)                # (B, n_classes)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            preds = torch.argmax(logits, dim=1)
            correct_train += (preds == yb).sum().item()
            total_train += yb.size(0)
        end_time_train_epoch = time.time()
        epoch_train_time = end_time_train_epoch-start_time_train_epoch

        train_loss = running_loss / total_train
        train_acc  = correct_train / total_train

        # Validation
        start_time_valid_epoch = time.time()
        val_loss, val_acc = eval_loader(model, valid_loader, criterion)
        end_time_valid_epoch = time.time()
        epoch_valid_time = end_time_valid_epoch- start_time_valid_epoch

        scheduler.step(val_acc)

        # Log history
        history['model'].append(i+1)
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['valid_loss'].append(val_loss)
        history['epoch_train_time'].append(epoch_train_time)
        history['epoch_valid_time'].append(epoch_valid_time)
        history['train_acc'].append(train_acc)
        history['valid_acc'].append(val_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        print(f"Model{i+1} Epoch{epoch:03d} | Train Loss:{train_loss:.4f} Acc:{train_acc:.4f} time:{epoch_train_time:.2f} s "
            f"| Val Loss:{val_loss:.4f} Acc:{val_acc:.4f} time:{epoch_valid_time:.2f}s  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Early stopping on val_loss
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(output_dir, f'best_inceptiontime_{i+1}.pth'))
            print(f"model {i+1} Saved best model at epoch {epoch}(val_acc={val_acc:.6f}) ")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"model {i+1} Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
            break


    end_time_train = time.time()
    total_train_time = end_time_train - start_time_train
    final_train_time += total_train_time
    print(f"Full training finished. Total training time of Model{i+1} : {total_train_time:.2f} seconds")




# trainset
y_train_true, y_train_pred = ensemble_evaluate(train_loader, n_models, device, desc="Train Set Inference")
train_acc_final = accuracy_score(y_train_true, y_train_pred)
train_f1_final = f1_score(y_train_true, y_train_pred, average='macro')

# valid
y_valid_true, y_valid_pred = ensemble_evaluate(valid_loader, n_models, device, desc="valid Set Inference")
valid_acc_final = accuracy_score(y_valid_true, y_valid_pred)
valid_f1_final = f1_score(y_valid_true, y_valid_pred, average='macro')

# test 
start_time_infer = time.time()
y_test_true, y_test_pred = ensemble_evaluate(test_loader, n_models, device, desc="Test Set Inference")
total_infer_time = time.time() - start_time_infer

test_acc_final = accuracy_score(y_test_true, y_test_pred)
test_f1_final = f1_score(y_test_true, y_test_pred, average='macro')
mean_01_loss = 1.0 - test_acc_final

report_dict = classification_report(y_test_true, y_test_pred, target_names=class_names, digits=4, output_dict=True)
report_text = classification_report(y_test_true, y_test_pred, target_names=class_names, digits=4)
cm = confusion_matrix(y_test_true, y_test_pred)

print(f"Final Accuracy on held-out test set: {test_acc_final:.4f}")
print(f"Final Test Set Mean 0-1 Loss (Error Rate): {mean_01_loss:.4f}")


#  Save outputs
# Classification report CSV
report_df = pd.DataFrame(report_dict).transpose()
report_csv_path = os.path.join(output_dir, 'classification_report_inceptiontime_final.csv')
report_df.to_csv(report_csv_path)
print(f"Classification report saved to {report_csv_path}")

# Confusion matrix PNG & NPY
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix on Test Set (Inceptiontime)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
confusion_matrix_path = os.path.join(output_dir, 'confusion_matrix_Inceptiontime_final.png')
plt.savefig(confusion_matrix_path)
plt.close()
cm_npy_path = os.path.join(output_dir, 'confusion_matrix_Inceptiontime_final_data.npy')
np.save(cm_npy_path, cm)
print(f"Confusion matrix saved to {confusion_matrix_path} & {cm_npy_path}")

# error analysis
misclassified_mask = y_test_pred != y_test_true
misclassified_indices = np.where(misclassified_mask)[0]

error_analysis_data = {
    'test_set_index': misclassified_indices,
    'true_label': y_test_true[misclassified_indices],
    'predicted_label': y_test_pred[misclassified_indices],
    'true_class_name': [class_names[i] for i in y_test_true[misclassified_indices]],
    'predicted_class_name': [class_names[i] for i in y_test_pred[misclassified_indices]]
}
error_df = pd.DataFrame(error_analysis_data)
error_csv_path = os.path.join(output_dir, 'error_analysis_inceptiontime_misclassified.csv')
error_df.to_csv(error_csv_path, index=False)



# Training log (per epoch)  CSV 
history_df = pd.DataFrame(history)
log_csv_path = os.path.join(output_dir, 'training_log_inceptiontime.csv')
history_df.to_csv(log_csv_path, index=False)



# Summary TXT 
summary_txt_path = os.path.join(output_dir, 'results_summary_inceptiontime.txt')
with open(summary_txt_path, 'a') as f:
    f.write(f"Full training finished. Total training time: {final_train_time:.4f} seconds \n")
    f.write(f"Final train Set Accuracy: {train_acc_final:.4f} F1 score :{train_f1_final:.4f}\n")
    f.write(f"Final valid Set Accuracy: {valid_acc_final:.4f} F1 score :{valid_f1_final:.4f}\n")
    f.write(f"Final Test Set Accuracy: {test_acc_final:.4f} F1 score :{test_f1_final:.4f}\n")
    f.write(f"Final Test Set Mean 0-1 Loss (Error Rate): {mean_01_loss:.4f} \n")
    f.write(f"Test Set Inference Time ({len(y_test_true)} Samples): {total_infer_time:.4f} s \n")
    f.write(f"Total Misclassified Samples: {len(error_df)}\n")
    f.write("--- Classification Report (Test Set) ---\n")
    f.write(report_text + "\n")
print(f"Summary saved to {summary_txt_path}")
