import logging
logging.basicConfig(level='ERROR')
import numpy as np
from tqdm import tqdm
import json
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
import matplotlib
import random
import os
import pandas as pd
import json


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# plot data 
def sweep(score, x):
    """
    Compute a ROC curve and then return the FPR, TPR, AUC, and ACC.
    """
    fpr, tpr, _ = roc_curve(x, -score)
    acc = np.max(1-(fpr+(1-tpr))/2)
    return fpr, tpr, auc(fpr, tpr), acc


def do_plot(prediction, answers, sweep_fn=sweep, metric='auc', legend="", output_dir=None, fpr_cap=0.05):
    """
    Generate the ROC curves by using ntest models as test models and the rest to train.
    """
    fpr_cap = fpr_cap
    fpr, tpr, auc, acc = sweep_fn(np.array(prediction), np.array(answers, dtype=bool))


    # Cap the FPR at the defined fpr_cap
    low = tpr[np.where(fpr<fpr_cap)[0][-1]]
    # bp()
    print(f'Attack %s   AUC %.4f, Accuracy %.4f, TPR@{fpr_cap*100}%%FPR of %.4f\n'%(legend, auc,acc, low))

    metric_text = ''
    if metric == 'auc':
        metric_text = 'auc=%.3f'%auc
    elif metric == 'acc':
        metric_text = 'acc=%.3f'%acc

    plt.plot(fpr, tpr, label=legend+metric_text)
    return legend, auc,acc, low

def fig_fpr_tpr(all_output, output_dir, fpr_cap):
    print("output_dir", output_dir)
    answers = []
    metric2predictions = defaultdict(list)
    
    for ex in all_output:
        answers.append(ex["label"])
        for metric in ex["pred"].keys():
            if ("raw" in metric) and ("clf" not in metric):
                continue
            metric2predictions[metric].append(ex["pred"][metric])
    
    plt.figure(figsize=(4,3))
    with open(f"{output_dir}/auc.txt", "w") as f:
        for metric, predictions in metric2predictions.items():
            legend, auc, acc, low = do_plot(predictions, answers, legend=metric,metric='auc', output_dir=output_dir)
            f.write(f'%s   AUC %.4f, Accuracy %.4f, TPR@{fpr_cap}%%FPR of %.4f\n'%(legend, auc, acc, low))

    plt.semilogx()
    plt.semilogy()
    plt.xlim(1e-5,1)
    plt.ylim(1e-5,1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot([0, 1], [0, 1], ls='--', color='gray')
    plt.subplots_adjust(bottom=.18, left=.18, top=.96, right=.96)
    plt.legend(fontsize=8)
    plt.savefig(f"{output_dir}/auc.png")

def fig_fpr_tpr_img(all_output, output_dir, fpr_cap, run_kl_metrics):
    examples_seen = 0
    fpr_cap = fpr_cap
    print(f"All Output Size: {len(all_output)}")
    print("output_dir", output_dir)
    
    # Output Dictionaries 
    partwise_metrics_dict = {}
    method_metrics = defaultdict(lambda: defaultdict(list))
    examplewise_metrics_dict = {}
    
    # Metrics Not Appropiate for AUC
    skipped_auc_metrics = ['Avg_non_normalised_kl_per_token','Full_renyi_05_Token','avg_kl_div_per_aug',
                           'avg_entropies_per_aug','Max_kl_per_token', 'All_ver_kl_per_token','token_regions',
                           'Avg_renyi_05_kl_per_token','Avg_renyi_1_kl_per_token','Avg_renyi_2_kl_per_token',
                           'Avg_renyi_inf_kl_per_token']
    
    # =======================================================================================================
    # PER EXAMPLE METRICS (Token Distributions, Per Token Values) (Not Sutable for AUC)
    # ========================================================================================================
    for ex in all_output:
        label = ex["label"]
        
        # For each method/part, store seperatly in examplewise_metrics_dict
        for method, preds in ex["pred"].items():
            
            if run_kl_metrics:
                if method not in examplewise_metrics_dict:
                    examplewise_metrics_dict[method] = {"Membership":{}, "Renyi 0.5 Entro per Tkn List":{}, "Across Tkn Avg Stnd Entro per Aug":{},"Across Tkn Avg KL-Div per Aug":{}, "Token Sequence Label": {}}
                    
                # Creating Row For said example
                row = {"Membership": label, "Renyi 0.5 Entro per Tkn List":np.nan, "Across Tkn Avg Stnd Entro per Aug":np.nan, "Across Tkn Avg KL-Div per Aug":np.nan, "Token Sequence Label":np.nan}
                
                full_renyi_05_token = preds.get("Full_renyi_05_Token") # Python List []
                avg_entropies_per_aug = preds.get("avg_entropies_per_aug") # Python Dict {'org_avg_entro': int, 'aug1_avg_entro': int, 'aug2_avg_entro': int,.....}
                token_regions = preds.get("token_regions")
                
                # TODO: Add all the renyi token KL distribution metrics
                
                # Formatting for JSON Dump
                if isinstance(full_renyi_05_token, np.ndarray):
                    full_renyi_05_token = full_renyi_05_token.tolist()
                avg_entropies_per_aug = {k: float(v) for k, v in avg_entropies_per_aug.items()}
                
                
                # Log the Full Renyi 0.5 Entropy Per Token List Per Example 
                if full_renyi_05_token is not None: row["Renyi 0.5 Entro per Tkn List"] = full_renyi_05_token
                
                # Log the Average Standard Entropy Per Augmentation Per Example
                if avg_entropies_per_aug is not None: row["Across Tkn Avg Stnd Entro per Aug"] = avg_entropies_per_aug
                
                # Log the label (img, inst, desp) the token came from
                if token_regions is not None: row['Token Sequence Label'] = token_regions
                
                        
                
                # =====================================================================
                        
            for metric, prediction in preds.items():
                if ("raw" in metric) and ("clf" not in metric):
                    continue
                
                # Handle KL-Divergence results seperatly
                if metric =='kl_divergence_results':
                    continue
                
                method_metrics[method][metric].append((prediction, label))
                
            
            if run_kl_metrics:
                # Handle All KL-Divergence Results
                all_normalised_kl_divergence_values_dict = preds.get("kl_divergence_results")    
                    #Include metrics (non_norm, renyi_05, renyi_1, renyi_2, renyi_inf):
                    # For each of these metrics, it includes: (augs_kl_divs_per_token, augs_kl_div_sum ,aug_kl_divs_avg_dict,
                    #                                       "Min_{ratio*100}% of Avg Kl_Div", "Min_{ratio*100}% of Max Kl_Div")
                    
                for normaliser, results in all_normalised_kl_divergence_values_dict.items():
                    # Variable Description: For every token, the average kl_divergence value across all augmentations
                    augs_avg_kl_divs_per_token = np.mean(results['augs_kl_divs_per_token'],axis=0)
                    if isinstance(augs_avg_kl_divs_per_token, np.ndarray):
                        augs_avg_kl_divs_per_token = augs_avg_kl_divs_per_token.tolist()
                        row[f'{normaliser}_normalised_avg_kl_per_token'] = augs_avg_kl_divs_per_token
                    # Variable Discription: Dictionary showing the average kl_divs for each aug
                        row[f'{normaliser}_normalised_avg_kl_divs_per_aug_dict'] = {k: float(v) for k, v in results['aug_kl_divs_avg_dict'].items()}
                        
                    # Variable Description: For each normalisation method, the average and max kl_divergence (used for scoring)
                    for ratio in [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                        avg = results[f"Min_{ratio*100}% of Avg Kl_Div"]
                        maxed = results[f"Min_{ratio*100}% of Max Kl_Div"]
                        
                        avg_title = f"{normaliser}_normalised_Min_{ratio*100}% of Avg Kl_Div"
                        max_title = f"{normaliser}_normalised_Min_{ratio*100}% of Max Kl_Div"
                        
                        method_metrics[method][avg_title].append((avg, label))
                        method_metrics[method][max_title].append((maxed, label))
                            
                for metric in row:
                    if metric not in examplewise_metrics_dict[method]:
                        examplewise_metrics_dict[method][metric] = {}
                    examplewise_metrics_dict[method][metric][examples_seen] = row[metric]   
                
        examples_seen += 1
    
    with open(f"{output_dir}/examplewise_additional_metrics.json", "w") as f:
        json.dump(examplewise_metrics_dict,f)
        
    
    # ======================================================================================================
    # 
    # ======================================================================================================
    
    
    
    # =======================================================================================================
    # Partwise Metrics (AUC, Acc, TPR) for all metrics each across all examples
    # ========================================================================================================
    for method, metrics in method_metrics.items():
        if method not in partwise_metrics_dict:
                partwise_metrics_dict[method] = {}
        
        for metric, data in metrics.items():
            if metric in skipped_auc_metrics:
                continue
            predictions, labels = zip(*data)
            legend, auc, acc, low = do_plot(predictions, labels, legend=metric, metric='auc', output_dir=None, fpr_cap=fpr_cap)
            
            partwise_metrics_dict[method][legend] = {'AUC':auc, 'Accuracy': acc, f'TPR@{fpr_cap*100}% FPR': low}
        
        plt.figure(figsize=(4,3))
        
        method_output_dir = f"{output_dir}/{method}"
        os.makedirs(method_output_dir, exist_ok=True)
        with open(f"{method_output_dir}/auc.txt", "w") as f:
            for metric, data in metrics.items():
                # If Metric raw value not score, skip
                if metric in skipped_auc_metrics:
                    continue
                
                predictions, labels = zip(*data)
                legend, auc, acc, low = do_plot(predictions, labels, legend=metric, metric='auc', output_dir=method_output_dir, fpr_cap=fpr_cap)
                f.write(f'{legend}   AUC {auc:.4f}, Accuracy {acc:.4f}, TPR@{fpr_cap*100}% FPR of {low:.4f}\n')

        plt.semilogx()
        plt.semilogy()
        plt.xlim(1e-5, 1)
        plt.ylim(1e-5, 1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.plot([0, 1], [0, 1], ls='--', color='gray')
        plt.subplots_adjust(bottom=.18, left=.18, top=.96, right=.96)
        plt.legend(fontsize=8)
        plt.savefig(f"{method_output_dir}/auc.png")
        plt.close()
        
    with open(f"{output_dir}/partwise_metrics.json", "w") as f:
        json.dump(partwise_metrics_dict,f)


def load_jsonl(input_path):
    with open(input_path, 'r') as f:
        data = [json.loads(line) for line in tqdm(f)]
    random.seed(0)
    random.shuffle(data)
    return data

def dump_jsonl(data, path):
    with open(path, 'w') as f:
        for line in tqdm(data):
            f.write(json.dumps(line) + "\n")

def read_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in tqdm(f)]

def convert_huggingface_data_to_list_dic(dataset):
    all_data = []
    for i in range(len(dataset)):
        ex = dataset[i]
        all_data.append(ex)
    
    random.shuffle(all_data)
    
    return all_data