import pandas as pd
import numpy as np

import pandas as pd
import re
import ast


def get_metric_token_array_per_example(path):
    full_array = []

    with open(path, "r") as file:
        lines = file.readlines()

    for line in lines:
            full_array.append(ast.literal_eval(line.split(":")[1].strip()))

    return pd.DataFrame(full_array)


def get_entropies(path):
    pattern = re.compile(r"\[org:\s*([\d.]+),\s*aug1:\s*([\d.]+),\s*aug2:\s*([\d.]+),\s*aug3:\s*([\d.]+),\s*aug4:\s*([\d.]+)\]")

    df = pd.DataFrame(columns=["org", "aug1", "aug2", "aug3", "aug4"])

    with open(path, "r") as file:
        for line in file:
            match = pattern.search(line)
            if match:
                values = list(map(float, match.groups()))
                df.loc[len(df)] = values

    return df

    

print("Converting: member_first_50_aug_avg_entropies_df")
member_first_50_aug_avg_entropies_df = get_entropies("/home/clo37/priv/VL-Large-MIA/results/image_MIA_fpr0.05_kl+renyi05+avgentro_full_tokens/img_Flickr/gen_32_tokens/img/member_first_50_aug_avg_entropies.txt")

print("Converting: member_first_50_avg_kl_df")
member_first_50_avg_kl_df = get_metric_token_array_per_example("/home/clo37/priv/VL-Large-MIA/results/image_MIA_fpr0.05_kl+renyi05+avgentro_full_tokens/img_Flickr/gen_32_tokens/img/member_first_50_avg_kl.txt")

print("Converting: member_first_50_full_renyi05_df")
member_first_50_full_renyi05_df = get_metric_token_array_per_example("/home/clo37/priv/VL-Large-MIA/results/image_MIA_fpr0.05_kl+renyi05+avgentro_full_tokens/img_Flickr/gen_32_tokens/img/member_first_50_full_renyi05.txt")

print("Converting: nonmember_first_50_aug_avg_entropies_df")
nonmember_first_50_aug_avg_entropies_df = get_entropies("/home/clo37/priv/VL-Large-MIA/results/image_MIA_fpr0.05_kl+renyi05+avgentro_full_tokens/img_Flickr/gen_32_tokens/img/nonmember_first_50_aug_avg_entropies.txt")

print("Converting: nonmember_first_50_avg_kl_df")
nonmember_first_50_avg_kl_df = get_metric_token_array_per_example("/home/clo37/priv/VL-Large-MIA/results/image_MIA_fpr0.05_kl+renyi05+avgentro_full_tokens/img_Flickr/gen_32_tokens/img/nonmember_first_50_avg_kl.txt")

print("Converting: nonmember_first_50_full_renyi05_df")
nonmember_first_50_full_renyi05_df = get_metric_token_array_per_example("/home/clo37/priv/VL-Large-MIA/results/image_MIA_fpr0.05_kl+renyi05+avgentro_full_tokens/img_Flickr/gen_32_tokens/img/nonmember_first_50_full_renyi05.txt")



nonmember_first_50_aug_avg_entropies_df.to_csv("nonmember_first_50_aug_avg_entropies.csv")
nonmember_first_50_avg_kl_df.to_csv("nonmember_first_50_avg_kl.csv")
nonmember_first_50_full_renyi05_df.to_csv("nonmember_first_50_full_renyi05.csv")

member_first_50_aug_avg_entropies_df.to_csv("member_first_50_aug_avg_entropies.csv")
member_first_50_avg_kl_df.to_csv("member_first_50_avg_kl.csv")
member_first_50_full_renyi05_df.to_csv("member_first_50_full_renyi05.csv")

















# fpr_05_kl_div_df = get_metric_token_array_per_example('/home/clo37/priv/VL-Large-MIA/analysis/first_50_avg_kl_fpr0.05.txt')
# fpr_05_renyi05_df = get_metric_token_array_per_example('/home/clo37/priv/VL-Large-MIA/results/image_MIA_fpr0.05_kl+renyi05+avgentro_full_tokens/img_Flickr/gen_32_tokens/img/first_50_full_renyi05.txt')
# fpr_05_aug_avg_entropies_df = get_entropies('/home/clo37/priv/VL-Large-MIA/results/image_MIA_fpr0.05_kl+renyi05+avgentro_full_tokens/img_Flickr/gen_32_tokens/img/first_50_aug_avg_entropies.txt')

# print("Full kl_div Token Array Dataframe:")
# print(fpr_05_kl_div_df)
# print('\n')

# #fpr_05_kl_div_df.to_csv("first_50_avg_kl_fpr0.05.csv")

# print("Full Renyi=0.5 Entropy Token Array Dataframe:")
# print(fpr_05_renyi05_df)
# print('\n')

# fpr_05_renyi05_df.to_csv("first_50_renyi05_fpr_05.csv")

# print("Perterbation Entries Dataframe:")
# print(fpr_05_aug_avg_entropies_df)
# print('\n')

# fpr_05_aug_avg_entropies_df.to_csv("first_50_peterb_entropies.csv")

