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

def fig_fpr_tpr_img(all_output, output_dir, fpr_cap):
    examples_seen = 0
    fpr_cap = fpr_cap
    
    print("output_dir", output_dir)
    method_metrics = defaultdict(lambda: defaultdict(list))
    method_tokenwise_kl = defaultdict(list)
    
    print(f"All Output Size: {len(all_output)}")
    
    out_dict = {}
        

    
    for ex in all_output:
        label = ex["label"]
        membership = 'nonmember'
        output_df = None
        
        if label == 1: membership = 'member'
        
        for method, preds in ex["pred"].items():
            
            if method not in out_dict:
                out_dict[method] = {"Membership":{},"Across Augs Avg KL-Div per Tkn List":{}, "Renyi 0.5 Entro per Tkn List":{}, "Across Tkn Avg Stnd Entro per Aug":{},"Across Tkn Avg KL-Div per Aug":{},}
                
            
            # method_output_dir = f"{output_dir}/{method}/"  # Ensure this is defined before writing
            # os.makedirs(method_output_dir, exist_ok=True)
            
            
            # ========================Additional Metrics========================================
            
            
            row = {"Membership": label,"Across Augs Avg KL-Div per Tkn List": np.nan, "Renyi 0.5 Entro per Tkn List":np.nan, "Across Tkn Avg Stnd Entro per Aug":np.nan, "Across Tkn Avg KL-Div per Aug":np.nan}
            
            avg_kl = preds.get("Avg_kl_per_token") #Python List []
            full_renyi_05_token = preds.get("Full_renyi_05_Token") # Python List []
            avg_entropies_per_aug = preds.get("avg_entropies_per_aug") # Python Dict {'org_avg_entro': int, 'aug1_avg_entro': int, 'aug2_avg_entro': int,.....}
            avg_kl_div_per_aug = preds.get("avg_kl_div_per_aug") # Python Dict {'org_avg_kl_div': 0, 'aug1_avg_kl_div': int, 'aug2_avg_kl_div': int,.....}
            
            
            # Formatting for JSON Dump
            if isinstance(avg_kl, np.ndarray):
                avg_kl = avg_kl.tolist()
            if isinstance(full_renyi_05_token, np.ndarray):
                full_renyi_05_token = full_renyi_05_token.tolist()
            avg_entropies_per_aug = {k: float(v) for k, v in avg_entropies_per_aug.items()}
            avg_kl_div_per_aug = {k: float(v) for k, v in avg_kl_div_per_aug.items()}
            
            # Log the Average KL Diveregence Per Token List Across all Examples
            if avg_kl is not None: row["Across Augs Avg KL-Div per Tkn List"] = avg_kl
            
            # Log the Full Renyi 0.5 Entropy Per Token List Per Example 
            if full_renyi_05_token is not None: row["Renyi 0.5 Entro per Tkn List"] = full_renyi_05_token
            
            # Log the Average Standard Entropy Per Augmentation Per Example
            if avg_entropies_per_aug is not None: row["Across Tkn Avg Stnd Entro per Aug"] = avg_entropies_per_aug
            
            # Log the Average KL Divergence Per Augmentation Per Example
            if avg_kl_div_per_aug is not None: row["Across Tkn Avg KL-Div per Aug"] = avg_kl_div_per_aug
            
            for metric in row:
                out_dict[method][metric][examples_seen] = row[metric]
            
            # Old Redudent Log Scripts
            '''
            # Write the average kl_divergence for the full token string into a txt
            if avg_kl is not None:
                kl_token_str = ", ".join([f"{v:.4f}" for v in avg_kl.tolist()])
                with open(f"{method_output_dir}/{membership}_first_50_avg_kl.txt", "a") as f:
                    f.write(f"For Example {examples_seen}, Average KL-DV of Tokens Across Perturbations for {method} method is: [{kl_token_str}]\n")
            
            # Write the renyo_05 entropy for the full token string into a txt
            if full_renyi_05_token is not None:
                renyi_05_token_str = ", ".join([f"{v:.4f}" for v in full_renyi_05_token])
                with open(f"{method_output_dir}/{membership}_first_50_full_renyi05.txt", "a") as f:
                    f.write(f"For Example {examples_seen}, The full renyi 0.5 token for {method} method is: [{renyi_05_token_str}]\n")
                    
            # Write the avg_entropies of tokens for the example original & Peterbation into txt
            if avg_entropies is not None:
                avg_entropies_str = ", ".join([f"{k}: {v:.4f}" for k, v in avg_entropies.items()])
                with open(f"{method_output_dir}/{membership}_avg_kl_div_per_aug.txt", "a") as f:
                    f.write(f"For Example {examples_seen}, average kl-divergence for augmentations across all token for {method} method is: [{avg_entropies_str}]\n")
            '''
            
            
            # =====================================================================
                    
            for metric, prediction in preds.items():
                if ("raw" in metric) and ("clf" not in metric):
                    continue
                method_metrics[method][metric].append((prediction, label))
                
        examples_seen += 1
    
    with open(f"{output_dir}/all_additional_metrics.json", "w") as f:
        json.dump(out_dict,f)
        
        
    # Metrics to Skip AUC Calculations as they are not doable
    skipped_auc_metrics = ['Avg_kl_per_token','Full_renyi_05_Token','avg_kl_div_per_aug','avg_entropies_per_aug']
                

    for method, metrics in method_metrics.items():
        method_output_dir = f"{output_dir}/{method}"
        os.makedirs(method_output_dir, exist_ok=True)
        

        plt.figure(figsize=(4,3))
        with open(f"{method_output_dir}/auc.txt", "w") as f:
            for metric, data in metrics.items():
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