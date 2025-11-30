import random
import torch
import numpy as np
import os
import shutil
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from utils import Dataset
from hydra_gpu import HydraGPU, HydraMultivariateGPU
from ridge import RidgeClassifier




SEED = 2025
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# use_cuda = torch.cuda.is_available()
# if not use_cuda:
#     print("No GPU detected. Please run this script on a machine with CUDA support.")
#     exit()

def get_predictions_hydra(classifier, data_loader):
    """ Helper function to get predictions batch by batch for RidgeClassifier """
    all_preds = []
    all_true = []
    for X, Y in tqdm(data_loader, desc="Predicting on test set"):
        pred_scores = classifier._predict(X) 
        all_preds.append(pred_scores.argmax(-1).cpu().numpy()) 
        all_true.append(Y) 
    return np.concatenate(all_preds), np.concatenate(all_true)

# param_grid_dict = {
#     'k': [12, 16, 20, 24],        
#     'g': [16, 32, 64],      
#     'batch_size': [256]  
    
# }

# param_grid = list(ParameterGrid(param_grid_dict))

# best_params = None
# best_cv_accuracy = 0.0
# best_cv_avg_train_time = 0.0
# cv_results = {}
# input_length = None

# load data 

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

dataset_dir = './processed_data'
output_dir = './processed_data/output'
x_train_path_full = os.path.join(dataset_dir, 'X_train_1d.npy')
y_train_path_full = os.path.join(dataset_dir, 'Y_train.npy')
x_test_path = os.path.join(dataset_dir, 'X_test_1d.npy')
y_test_path = os.path.join(dataset_dir, 'Y_test.npy')

X_train_full = np.load(x_train_path_full)
Y_train_full = np.load(y_train_path_full)
print("Loading training data is done")


input_length = X_train_full.shape[2]
print(f"Input length for HYDRA: {input_length}")


# for i, params in enumerate(param_grid):
#     print(f"\n{'-'*20} Param combination {i+1}/{len(param_grid)}: {params} {'-'*20}")
#     n_splits = 5
#     skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
#     cv_val_accuracies = []
#     cv_train_accuracies = []
#     cv_train_times = []

#     # --- [HYDRA] 
#     if input_length is None:
#         # X_train_full shape [N, C, L] -> [60000, 1, 1024]
#         input_length = X_train_full.shape[2] 
#         print(f"input_length for HYDRA: {input_length}")

#     current_batch_size = params.get('batch_size', 256)

#     for fold, (train_idx, val_idx) in enumerate(tqdm(skf.split(X_train_full, Y_train_full), total=n_splits, desc="CV progress")):
        
#         X_train_fold, X_val_fold = X_train_full[train_idx], X_train_full[val_idx]
#         Y_train_fold, Y_val_fold = Y_train_full[train_idx], Y_train_full[val_idx]

# 
#         temp_dir = f'./temp_cv_data_params_hydra_{i}_fold_{fold}' 
#         os.makedirs(temp_dir, exist_ok=True)
        
#         path_x_train_fold = os.path.join(temp_dir, 'X_train_fold.npy')
#         path_y_train_fold = os.path.join(temp_dir, 'Y_train_fold.npy')
#         path_x_val_fold = os.path.join(temp_dir, 'X_val_fold.npy')
#         path_y_val_fold = os.path.join(temp_dir, 'Y_val_fold.npy')
        
#         np.save(path_x_train_fold, X_train_fold)
#         np.save(path_y_train_fold, Y_train_fold)
#         np.save(path_x_val_fold, X_val_fold)
#         np.save(path_y_val_fold, Y_val_fold)
        

#         # [HYDRA] 
#         train_fold_data = Dataset(path_x_train_fold, path_y_train_fold, batch_size=current_batch_size, shuffle=False)
#         val_fold_data = Dataset(path_x_val_fold, path_y_val_fold, batch_size=current_batch_size, shuffle=False)
        
#         # 1. 创建 HYDRA 特征转换器
#         hydra_transform = HydraGPU(
#             input_length=input_length,
#             k=params['k'],
#             g=params['g'],
#             seed=SEED
#         ).to(device) # <-- 发送到 GPU

#         # 2. 创建 Ridge 分类器，并将转换器和设备传给它
#         classifier_fold = RidgeClassifier(
#             transform=hydra_transform,
#             device=device,
#             verbose=False 
#         )
#       
        
#         start_time = time.time()
#         classifier_fold.fit(train_fold_data) # fit 接收分批的数据
#         end_time = time.time()
#         fold_train_time = end_time - start_time
#         cv_train_times.append(fold_train_time)
        
#         # === [HYDRA] 6. 使用 .score() (无需 .unbatch()) ===
#         # RidgeClassifier.score() 期望分批的数据加载器
#         train_error_rate = classifier_fold.score(train_fold_data).item()
#         train_accuracy = 1 - train_error_rate
#         cv_train_accuracies.append(train_accuracy)

#         val_error_rate = classifier_fold.score(val_fold_data).item()
#         val_accuracy = 1 - val_error_rate
#         cv_val_accuracies.append(val_accuracy)
#         

#         train_fold_data.close()
#         val_fold_data.close()
#        
#         try:
#             shutil.rmtree(temp_dir)
#         except OSError as e:
#             print(f"Warning: Could not remove temp dir {temp_dir}: {e}")

#     # --- CV result
#     mean_cv_val_accuracy = np.mean(cv_val_accuracies)
#     std_cv_val_accuracy = np.std(cv_val_accuracies)
#     mean_cv_train_accuracy = np.mean(cv_train_accuracies)
#     mean_cv_train_time = np.mean(cv_train_times)
        
#     cv_results[str(params)] = {
#         'val_accuracy_mean': mean_cv_val_accuracy, 
#         'val_accuracy_std': std_cv_val_accuracy,
#         'train_accuracy_mean': mean_cv_train_accuracy,
#         'train_time_mean': mean_cv_train_time,
#         'val_fold_scores': cv_val_accuracies 
#     }

#     if mean_cv_val_accuracy > best_cv_accuracy:
#         best_cv_accuracy = mean_cv_val_accuracy
#         best_params = params
#         best_cv_avg_train_time = mean_cv_train_time


# print(f"\n Best Params : {best_params} (accuracy-val: {best_cv_accuracy:.4f})")
# print("--------------------------------------------------")


# df_results = pd.DataFrame.from_dict(cv_results, orient='index')
# fold_scores_df = pd.DataFrame(df_results['val_fold_scores'].tolist(), index=df_results.index)
# fold_scores_df = fold_scores_df.rename(columns={i: f'fold_{i+1}_score' for i in range(n_splits)})
# df_results = pd.concat([df_results.drop('val_fold_scores', axis=1), fold_scores_df], axis=1)
# csv_path = os.path.join(output_dir, 'cv_results_detailed_hydra.csv')
# df_results.to_csv(csv_path)
# print(f"CV results saved to {csv_path}")

best_params = {
    'k': 28,        
    'g': 64,      
    'batch_size': 256 
}

print("\nTraining final HYDRA model using best parameters...")
best_batch_size = best_params.get('batch_size', 256)

training_data_full = Dataset(x_train_path_full, y_train_path_full, batch_size=best_batch_size, shuffle=False)

final_hydra_transform = HydraGPU(
    input_length=input_length,
    k=best_params['k'],
    g=best_params['g'],
    seed=SEED
).to(device)

final_classifier = RidgeClassifier(
    transform=final_hydra_transform,
    device=device,
    verbose=True
)

start_time = time.time()
final_classifier.fit(training_data_full)
end_time = time.time()
final_train_time = end_time - start_time
training_data_full.close()
print(f"Final model training finished in {final_train_time:.2f} seconds.")



train_eval_data = Dataset(x_train_path_full, y_train_path_full, batch_size=best_batch_size, shuffle=False)
y_train_pred, y_train_true = get_predictions_hydra(final_classifier, train_eval_data)
train_eval_data.close()

# 1. Train Classification Report ( Precision, Recall, F1)
train_report_dict = classification_report(y_train_true, y_train_pred, target_names=class_names, digits=4, output_dict=True)
train_report_df = pd.DataFrame(train_report_dict).transpose()
train_report_df.to_csv(os.path.join(output_dir, 'classification_report_hydra_train.csv'))

# 2. Train Confusion Matrix
cm_train = confusion_matrix(y_train_true, y_train_pred)

# Train Confusion Matrix 
plt.figure(figsize=(12, 10))
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix on TRAINING Set (HYDRA)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'confusion_matrix_hydra_train.png'))
np.save(os.path.join(output_dir, 'confusion_matrix_hydra_train.npy'), cm_train)
plt.close()


# final test
print("\nEvaluating final model on the test set...")
# using 'Dataset'
testing_data = Dataset(x_test_path, y_test_path, batch_size=best_batch_size, shuffle=False)
# y_test_true = np.load(y_test_path) # 

start_time = time.time()
#  get_predictions_hydra
y_test_pred, y_test_true = get_predictions_hydra(final_classifier, testing_data)
end_time = time.time()
inference_time = end_time - start_time
testing_data.close()


test_accuracy = accuracy_score(y_test_true, y_test_pred)
mean_01_loss =  1 - test_accuracy

report = classification_report(y_test_true, y_test_pred, digits=4, output_dict=True)
cm = confusion_matrix(y_test_true, y_test_pred)


print("\n--- Final Model Results (HYDRA) ---")
print(f"Best Hyperparameters Used: {best_params}")
print(f"Final Model Training Time (Full Data): {final_train_time:.4f} s")
print(f"Final Test Set Accuracy: {test_accuracy:.4f}")
print(f"Final Test Set Mean 0-1 Loss (Error Rate): {mean_01_loss:.4f}")
print(f"Test Set Inference Time ({len(y_test_true)} Samples, batch_size {best_batch_size}): {inference_time:.4f} s")
print("\nClassification Report on Test Set:")
report_text = classification_report(y_test_true, y_test_pred, target_names=class_names, digits=4)
print(report_text)


report_df = pd.DataFrame(report).transpose()
report_csv_path = os.path.join(output_dir, 'classification_report_hydra_final.csv')
report_df.to_csv(report_csv_path)
print(f"Classification report saved to {report_csv_path}")


plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix on Test Set (HYDRA)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
confusion_matrix_path = os.path.join(output_dir, 'confusion_matrix_hydra_final.png')
plt.savefig(confusion_matrix_path)
confusion_matrix_data_path = os.path.join(output_dir, 'confusion_matrix_hydra_final_data.npy')
np.save(confusion_matrix_data_path, cm)
plt.close()

###### error analysis
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


error_csv_path = os.path.join(output_dir, 'error_analysis_hydra_misclassified.csv')
error_df.to_csv(error_csv_path, index=False)


summary_txt_path = os.path.join(output_dir, 'results_summary_hydra.txt')
print(f"\nSaving summary results to {summary_txt_path}...")
with open(summary_txt_path, 'a') as f: 
    f.write(f"Best Hyperparameters Used: {best_params}\n")
    f.write(f"Final Model Training Time (Full Data): {final_train_time:.4f} s\n") 
    f.write(f"Final Test Set Accuracy: {test_accuracy:.4f}\n")
    f.write(f"Final Test Set Mean 0-1 Loss (Error Rate): {mean_01_loss:.4f}\n")
    f.write(f"Test Set Inference Time ({len(y_test_true)} Samples, batch_size {best_batch_size}): {inference_time:.4f} s\n") 
    f.write(f"Total misclassified samples: {len(error_df)} \n")
print("Summary results saved.")


