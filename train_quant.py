import numpy as np
from quant import QuantClassifier
from utils import BatchDataset
import os
import time
from tqdm import tqdm
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import json
import scipy.stats as stats

import random
SEED = 2025
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)



def get_predictions(classifier, data_loader):
    all_preds = []
    for X, _ in tqdm(data_loader, desc="Predicting"): 
        Z = classifier.transform.transform(torch.tensor(X.astype(np.float32)))
        preds = classifier.classifier.predict(Z)
        all_preds.append(preds)
    return np.concatenate(all_preds)


# param_grid_dict = {
#     'num_estimators':[200],
#     'depth': [12,14],
#     'div':[8],
#     'limit_mb':[256]
# }
# param_grid = list(ParameterGrid(param_grid_dict))

# best_params = None
# best_cv_accuracy = 0.0
# best_cv_avg_train_time = 0.0
# cv_results = {}

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

dataset_dir = './processed_data'
output_dir = './processed_data/output/quant'
x_train_path_full = os.path.join(dataset_dir, 'X_train_1d.npy')
y_train_path_full = os.path.join(dataset_dir, 'Y_train.npy')
x_test_path = os.path.join(dataset_dir, 'X_test_1d.npy')
y_test_path = os.path.join(dataset_dir, 'Y_test.npy')

# X_train_full = np.load(x_train_path_full)
# Y_train_full = np.load(y_train_path_full)
# print("Loading training data is done")

# for i, params in enumerate(param_grid):
#     print(f"\n{'-'*20} Param combination {i+1}/{len(param_grid)}: {params} {'-'*20}")
#     n_splits = 5
#     skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
#     cv_val_accuracies = []
#     cv_train_accuracies = []
#     cv_train_times = []

#     for fold, (train_idx, val_idx) in enumerate(tqdm(skf.split(X_train_full, Y_train_full), total=n_splits, desc="CV progress")):
        
#         X_train_fold, X_val_fold = X_train_full[train_idx], X_train_full[val_idx]
#         Y_train_fold, Y_val_fold = Y_train_full[train_idx], Y_train_full[val_idx]

#         temp_dir = f'./temp_cv_data_params_{i}'
#         os.makedirs(temp_dir, exist_ok=True)
        
#         path_x_train_fold = os.path.join(temp_dir, 'X_train_fold.npy')
#         path_y_train_fold = os.path.join(temp_dir, 'Y_train_fold.npy')
#         path_x_val_fold = os.path.join(temp_dir, 'X_val_fold.npy')
#         path_y_val_fold = os.path.join(temp_dir, 'Y_val_fold.npy')
        
#         np.save(path_x_train_fold, X_train_fold)
#         np.save(path_y_train_fold, Y_train_fold)
#         np.save(path_x_val_fold, X_val_fold)
#         np.save(path_y_val_fold, Y_val_fold)

#         train_fold_data = BatchDataset(path_x_train_fold, path_y_train_fold)
#         val_fold_data = BatchDataset(path_x_val_fold, path_y_val_fold,shuffle=False)
#         # train_fold_data = BatchDataset(X_train_fold, Y_train_fold)
#         # val_fold_data = BatchDataset(X_val_fold, Y_val_fold, shuffle=False)
        
#         classifier_fold = QuantClassifier(verbose=False, **params)
        
#         start_time = time.time()
#         classifier_fold.fit(train_fold_data)
#         end_time = time.time()
#         fold_train_time = end_time - start_time
#         cv_train_times.append(fold_train_time)
        
#         train_fold_data_unbatched = train_fold_data.unbatch() 
#         train_error_rate = classifier_fold.score(train_fold_data_unbatched)
#         train_accuracy = 1 - train_error_rate
#         cv_train_accuracies.append(train_accuracy)

#         val_fold_data_unbatched = val_fold_data.unbatch()
#         val_error_rate = classifier_fold.score(val_fold_data_unbatched)
#         val_accuracy = 1 - val_error_rate
#         cv_val_accuracies.append(val_accuracy)

#         train_fold_data.close()
#         val_fold_data.close()
#         if hasattr(train_fold_data_unbatched, 'close'): 
#             train_fold_data_unbatched.close()
#         if hasattr(val_fold_data_unbatched, 'close'): 
#             val_fold_data_unbatched.close()
        

#         # temp_files = [path_x_train_fold, path_y_train_fold, path_x_val_fold, path_y_val_fold]
#         # for f_path in temp_files:
#         #     if os.path.exists(f_path):
#         #         os.remove(f_path)
        
#         # if os.path.exists(temp_dir) and not os.listdir(temp_dir):
#         #     os.rmdir(temp_dir)

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

# sorted_results = sorted(cv_results.items(), key=lambda item: item[1]['val_accuracy_mean'], reverse=True)
# # for params_str, result in sorted_results:
# #     print(f"Params: {params_str}, val_accuracy_mean = {result['val_accuracy_mean']:.4f}")
# print(f"\n Best Params : {best_params} (accuracy-val: {best_cv_accuracy:.4f})")
# print("--------------------------------------------------")

# # significant testing
# if len(sorted_results) >= 2:
#     best_config_str, best_result = sorted_results[0]
#     second_best_config_str, second_best_result = sorted_results[1]

#     best_scores = best_result['val_fold_scores']
#     second_best_scores = second_best_result['val_fold_scores']

#     stat, p_value = stats.wilcoxon(best_scores, second_best_scores)
    
#     print(f"\n Wilcoxon results: p-value = {p_value:.4f}")
    
#     alpha = 0.05
#     if p_value < alpha:
#         print(f"Wilcoxon results: {p_value:.4f} < {alpha}, statistically significant")
#     else:
#         print(f"Wilcoxon results: {p_value:.4f} > {alpha}, not significant")
# else:
#     print("resutls < 2, no  significant testing")
# print("--------------------------------------------------")


# df_results = pd.DataFrame.from_dict(cv_results, orient='index')
# fold_scores_df = pd.DataFrame(df_results['val_fold_scores'].tolist(), index=df_results.index)
# fold_scores_df = fold_scores_df.rename(columns={i: f'fold_{i+1}_score' for i in range(n_splits)})
# df_results = pd.concat([df_results.drop('val_fold_scores', axis=1), fold_scores_df], axis=1)
# csv_path = os.path.join(output_dir, 'cv_results_detailed_quant_depth_div_1.csv')
# df_results.to_csv(csv_path)
# print("csv is saved")

best_params = {'depth': 10, 'div': 8, 'limit_mb': 256, 'num_estimators': 200, 'random_state': SEED}
print(f"Using best parameters: {best_params}")

training_data_full = BatchDataset(x_train_path_full, y_train_path_full)
final_classifier = QuantClassifier(verbose=True, **best_params)

start_time = time.time()
final_classifier.fit(training_data_full)
end_time = time.time()
final_train_time = end_time - start_time
training_data_full.close()

# geting training set accuracy classification metrics and confusion 
train_eval_data = BatchDataset(x_train_path_full, y_train_path_full,shuffle=False)
y_train_pred = get_predictions(final_classifier, train_eval_data)
train_eval_data.close()
y_train_true = np.load(y_train_path_full)

# 1. Train Classification Report ( Precision, Recall, F1)
train_report_dict = classification_report(y_train_true, y_train_pred, target_names=class_names, digits=4, output_dict=True)
train_report_df = pd.DataFrame(train_report_dict).transpose()
train_report_df.to_csv(os.path.join(output_dir, 'classification_report_hydra_train.csv'))

# 2. Train Confusion Matrix
cm_train = confusion_matrix(y_train_true, y_train_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix on TRAINING Set (QUANT)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'confusion_matrix_quant_train.png'))
np.save(os.path.join(output_dir, 'confusion_matrix_quant_train.npy'), cm_train)
plt.close()


testing_data = BatchDataset(x_test_path, y_test_path, shuffle=False)
testing_limit_mb = best_params['limit_mb']
testing_data.set_batch_size(testing_limit_mb)  # using it after cv period
# testing_data_unbatched = testing_data.unbatch()
y_test_true = np.load(y_test_path)

print(f"Diagnostic Check: Length of y_test_true is {len(y_test_true)}")
# print(f"Diagnostic Check: Number of examples in testing_data loader is {len(testing_data)}")

start_time = time.time()
y_test_pred = get_predictions(final_classifier, testing_data)
# test_error_rate = final_classifier.score(testing_data_unbatched)
end_time = time.time()
inference_time = end_time - start_time
testing_data.close()
# testing_data_unbatched.close() 

# print(f"Diagnostic Check: Number of predictions generated is {len(y_test_pred)}")


test_accuracy = accuracy_score(y_test_true, y_test_pred)
mean_01_loss =  1 - test_accuracy

report = classification_report(y_test_true, y_test_pred, digits=4, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_csv_path = os.path.join(output_dir, 'classification_report_quant_final.csv')
report_df.to_csv(report_csv_path)
print(f"Classification report saved to {report_csv_path}")

# test_accuracy = 1 - test_error_rate
# print(f"Best Hyperparameter Combination: {best_params}")
# print(f"Cross-Validation Mean Accuracy (Validation Set): {best_cv_accuracy:.4f}")
# print(f"Final Test Set Accuracy: {test_accuracy:.4f} (or {test_accuracy:.2%})")
# print(f"Final Test Set Mean 0-1 Loss (Error Rate): {mean_01_loss:.4f}")
# print(f"Average Fold Training Time (CV): {best_cv_avg_train_time:.2f} s")
# print(f"Final Model Training Time: {final_train_time:.2f} s")
# print(f"Test Set Inference Time (10000 Samples): {inference_time:.2f} s")


# # model size
# model_path = os.path.join(output_dir, 'final_quant_model-.joblib')
# joblib.dump(final_classifier, model_path)
# model_size_bytes = os.path.getsize(model_path)
# model_size_kb = model_size_bytes / 1024

# confusion matrix
cm = confusion_matrix(y_test_true, y_test_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix on Test Set')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
confusion_matrix_path = os.path.join(output_dir, 'confusion_matrix_quant.png')
plt.savefig(confusion_matrix_path)
confusion_matrix_data_path = os.path.join(output_dir, 'confusion_matrix_quant_final_data.npy')
np.save(confusion_matrix_data_path, cm)
print(f"Confusion matrix raw data saved to {confusion_matrix_data_path}")

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


error_csv_path = os.path.join(output_dir, 'error_analysis_quant_misclassified.csv')
error_df.to_csv(error_csv_path, index=False)


summary_txt_path = os.path.join(output_dir, 'results_summary_quant.txt')
print(f"\nSaving summary results to {summary_txt_path}...")
with open(summary_txt_path, 'a') as f: # using "a"
    f.write(f"Best Hyperparameters Used: {best_params}\n")
    f.write(f"Final Model Training Time (Full Data): {final_train_time:.4f} s\n") 
    f.write(f"Final Test Set Accuracy: {test_accuracy:.4f}\n")
    f.write(f"Final Test Set Mean 0-1 Loss (Error Rate): {mean_01_loss:.4f}\n")
    f.write(f"Test Set Inference Time ({len(y_test_true)} Samples, batch size {testing_limit_mb}): {inference_time:.4f} s\n") 
    f.write(f"Total misclassified samples: {len(error_df)} \n")
    f.write("\n\n") 
print("Summary results saved.")