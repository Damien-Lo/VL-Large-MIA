import pandas as pd
import numpy as np

import pandas as pd
import re



def extract_metrics(path):
    
    with open(path, "r") as file:
        lines = file.readlines()
    
    pattern = re.compile(r"(.+?)\s+AUC\s+([0-9.]+), Accuracy ([0-9.]+), TPR@([0-9.]+)% FPR of ([0-9.]+)")
    FPR = None
    
    data=[]
    
    for line in lines:    
        match = pattern.match(line)
        
        if match:
            metric = match.group(1).strip()
            auc = float(match.group(2))
            acc = float(match.group(3))
            fpr = float(match.group(4))
            tpr = float(match.group(5))
            data.append((metric, auc, acc, tpr))
            
            if FPR is None:
                FPR = fpr
            elif fpr != FPR:
                print(f"Warning: Line has a different FPR ({fpr} vs {FPR})")
            
    df = pd.DataFrame(data, columns=["Metric", "AUC", "Accuracy", f"TPR@{FPR}%FPR"])

    return df


fpr_one_percent = extract_metrics('/home/clo37/priv/VL-Large-MIA/cluster_scripts/image_MIA_fpr0.01/img_Flickr/gen_32_tokens/img/auc.txt')
fpr_five_percent = extract_metrics('/home/clo37/priv/VL-Large-MIA/cluster_scripts/image_MIA_fpr0.05/img_Flickr/gen_32_tokens/img/auc.txt')

fpr_one_percent.to_csv('metics_fpr_one_percent.csv')
fpr_five_percent.to_csv('metics_fpr_five_percent.csv')








# df_fpr5 = extract_tpr(fpr=5,file_path= '/local/scratch/clo37/vlm_large_mia/VL-Large-MIA/cluster_scripts/image_MIA/img_Flickr/gen_32_tokens/img/auc.txt')
# df_fpr1 = extract_tpr(fpr=1,file_path= '/local/scratch/clo37/vlm_large_mia/VL-Large-MIA/cluster_scripts/image_MIA_fpr0.01/img_Flickr/gen_32_tokens/img/auc.txt')
# df_fpr001 = extract_tpr(fpr=0.1,file_path='/local/scratch/clo37/vlm_large_mia/VL-Large-MIA/cluster_scripts/image_MIA_fpr0.001/img_Flickr/gen_32_tokens/img/auc.txt')

# df = pd.merge(df_fpr5, df_fpr1, on="Metric", how="outer")
# df = pd.merge(df, df_fpr001, on="Metric", how="outer")

# select_metrics = ['ppl','Min_0% Prob','Min_10.0% Prob','Min_20.0% Prob','aug_kl','Max_Prob_Gap','Modified_renyi_05',
#                   'Modified_entropy','Modified_renyi_2',
#                   'Max_0% renyi_05','Max_10.0% renyi_05','Max_100% renyi_05',
#                   'Max_0% renyi_1','Max_10.0% renyi_1','Max_100% renyi_1',
#                   'Max_0% renyi_2','Max_10.0% renyi_2','Max_100% renyi_2',
#                   'Max_0% renyi_inf','Max_10.0% renyi_inf','Max_100% renyi_inf']

# df_summary = df[df["Metric"].apply(lambda x: x in select_metrics)].copy()

# print("Full DF \n")
# print(df)
# print()

# print("Summary DF \n")
# print(df_summary)





