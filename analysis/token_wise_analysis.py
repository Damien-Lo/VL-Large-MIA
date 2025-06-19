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


fpr_05_kl_div_df = get_metric_token_array_per_example('/home/clo37/priv/VL-Large-MIA/analysis/first_50_avg_kl_fpr0.05.txt')
fpr_05_renyi05_df = get_metric_token_array_per_example('/home/clo37/priv/VL-Large-MIA/results/image_MIA_fpr0.05_kl+renyi05_full_tokens/img_Flickr/gen_32_tokens/img/first_50_full_renyi05.txt')

print("Full kl_div Token Array Dataframe:")
print(fpr_05_kl_div_df)
print('\n')

#fpr_05_kl_div_df.to_csv("first_50_avg_kl_fpr0.05.csv")

print("Full Renyi=0.5 Entropy Token Array Dataframe:")
print(fpr_05_renyi05_df)
print('\n')

fpr_05_renyi05_df.to_csv("first_50_renyi05_fpr_05.csv")

