import numpy as np
import zlib
from scipy.stats import entropy
import statistics
import json
import pdb
import torch
from PIL import Image

def get_text_metric(ppl, all_prob, p1_likelihood, entropies, mod_entropy, max_p, org_prob, gap_p, renyi_05, renyi_2, text, ppl_lower, mod_renyi_05, mod_renyi_2):
    pred = {}

    zlib_entropy = len(zlib.compress(bytes(text, 'utf-8')))
    
    pred["ppl"] = ppl
    pred["ppl/zlib"] = np.log(ppl)/zlib_entropy
    pred["ppl/lowercase_ppl"] = - (np.log(ppl_lower) / np.log(ppl)).item()

    # mink
    for ratio in [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        k_length = int(len(all_prob)*ratio)
        if k_length == 0:
            k_length = 1
        topk_prob = np.sort(all_prob)[:k_length]
        pred[f"Min_{ratio*100}% Prob"] = -1* np.mean(topk_prob).item()

    pred["Modified_entropy"] = np.nanmean(mod_entropy).item()

    pred["Modified_renyi_05"] = np.nanmean(mod_renyi_05).item()

    pred["Modified_renyi_2"] = np.nanmean(mod_renyi_2).item()

    pred["Max_Prob_Gap"] = -np.mean(gap_p).item()

    for ratio in [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        k_length = int(len(renyi_05)*ratio)
        if k_length == 0:
            k_length = 1
        topk_prob = np.sort(renyi_05)[-k_length:]
        pred[f"Max_{ratio*100}% renyi_05"] = np.mean(topk_prob).item()
        topk_prob = np.sort(entropies)[-k_length:]
        pred[f"Max_{ratio*100}% renyi_1"] = np.mean(topk_prob).item()
        topk_prob = np.sort(renyi_2)[-k_length:]
        pred[f"Max_{ratio*100}% renyi_2"] = np.mean(topk_prob).item()
        topk_prob = np.sort(-np.array(max_p))[-k_length:]
        pred[f"Max_{ratio*100}% renyi_inf"] = np.mean(topk_prob).item()


    for ratio in [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        k_length = int(len(renyi_05)*ratio)
        if k_length == 0:
            k_length = 1
        topk_prob = np.sort(renyi_05)[:k_length]
        pred[f"Min_{ratio*100}% renyi_05"] = np.mean(topk_prob).item()
        topk_prob = np.sort(entropies)[:k_length]
        pred[f"Min_{ratio*100}% renyi_1"] = np.mean(topk_prob).item()
        topk_prob = np.sort(renyi_2)[:k_length]
        pred[f"Min_{ratio*100}% renyi_2"] = np.mean(topk_prob).item()
        topk_prob = np.sort(-np.array(max_p))[:k_length]
        pred[f"Min_{ratio*100}% renyi_inf"] = np.mean(topk_prob).item()

    return pred

def kl_divergence(p, log_p, log_q):
    kl_div = np.sum(p * (log_p - log_q))
    return kl_div

def kl_divergence_per_token(p, log_p, log_q):
    return np.sum(p*(log_p-log_q),axis=1)

def cross_entropy(p,log_q):
    return -np.sum(p*log_q)

def cross_entropy_per_token(p,log_q):
    return -np.sum(p*log_q,axis=1)

def renyi_divergence_per_token(p, q, alpha, eps=1e-12):
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    divergence = (1 / (alpha - 1)) * np.log(np.sum(p**alpha * q**(1 - alpha), axis=1) + eps)
    return divergence



def get_img_metric(ppl, all_prob, p1_likelihood, entropies, mod_entropy, max_p, org_prob, gap_p, renyi_05_entro, renyi_2_entro, log_probs, mod_renyi_05, mod_renyi_2,
                    org_cross_entro_per_token, augmented_images_CE_per_token, all_aug_metrics, transformation_keys, original_probabilties_dict):
    
    pred = {}
    
    # # Convert Each Element to Float
    org_cross_entro_per_token = np.array([t.item() for t in org_cross_entro_per_token])
    
    # ======= Cross Entropy Loss ================
    for ratio in [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:        
        avg_entro_loss_per_aug = []
        k_length = int(len(org_cross_entro_per_token)*ratio)
        if k_length == 0:
            k_length = 1
        
        for aug in augmented_images_CE_per_token:
            
            aug_max_entros_idx = np.argsort(aug)[-k_length:]      # Get the indecies of the largest entropy losses of augmentation
            aug_max_entros_avg = aug[aug_max_entros_idx].mean()
            org_max_entros_avg = org_cross_entro_per_token[aug_max_entros_idx].mean()
            
            loss = np.abs(org_max_entros_avg - aug_max_entros_avg)
            avg_entro_loss_per_aug.append(loss)
            
        
        pred[f"Min_{ratio*100}% Cross_Entro_Augs"] = -1* np.mean(avg_entro_loss_per_aug)
           

    # ======================================= KL Divergence Start ================================================
    all_normalised_kl_divergence_values_dict = {}
    eps = 1e-12
    for metric, base_probs in original_probabilties_dict.items():
        all_normalised_kl_divergence_values_dict[metric] = {'augs_kl_divs_per_token': [], 'augs_kl_div_sum': []} # Where each [] holds results for each aug
    
    for metric, base_probs in original_probabilties_dict.items():
        # print(f"Working on metric {metric}")
        count = 1
        for aug_metrics in all_aug_metrics:
            aug_kl_divs_per_token = [] # (num_vers, seq_len)
            aug_kl_div_sum = [] 
            # print(f"Augmentation {count}")
            for version_metric in aug_metrics:
                version_log_probs = (
                    version_metric['log_probs'] if metric == 'no_norm' else torch.log(version_metric[metric] + eps)
                )
                
                if isinstance(base_probs, list):
                    print("base_probs is a list, converting to torchtensor")
                    base_probs = torch.tensor(base_probs)
                if isinstance(version_log_probs, np.ndarray):
                    print("version_log_probs is a nparray, converting to torchtensor")
                    version_log_probs = torch.tensor(version_log_probs)
                    
                # print(f"base probs has type: {type(base_probs)} and shape: {base_probs.shape}")
                # print(f"version_log_probs has type: {type(version_log_probs)} and shape: {version_log_probs.shape}")
                
                kl = kl_divergence(base_probs.cpu().numpy(), torch.log(base_probs+eps).cpu().numpy(), version_log_probs.cpu().numpy()).mean()
                kl_per_token = kl_divergence_per_token(base_probs.cpu().numpy(), torch.log(base_probs+eps).cpu().numpy(), version_log_probs.cpu().numpy())
                
                aug_kl_divs_per_token.append(kl_per_token)
                aug_kl_div_sum.append(kl)
                
            all_normalised_kl_divergence_values_dict[metric]['augs_kl_divs_per_token'].append(np.mean(aug_kl_divs_per_token, axis=0))
            all_normalised_kl_divergence_values_dict[metric]['augs_kl_div_sum'].append(np.mean(aug_kl_div_sum))
            count += 1
    
    # Defining average kl for each type of augmentation for each normalisation method
    for metric, storage in all_normalised_kl_divergence_values_dict.items():
        aug_kl_divs_avg_dict = {'org_avg_kl_div': 0}
        title = f'avg_kl_div_per_aug_{str(metric)}_normalised'
        for i in range(len(transformation_keys)):
            aug_kl_divs_avg_dict[transformation_keys[i]] = storage['augs_kl_div_sum'][i]
        
        all_normalised_kl_divergence_values_dict[metric]['aug_kl_divs_avg_dict'] = aug_kl_divs_avg_dict
       
       
        # Mink of Averag tokens or max per token 
        all_per_token_kl_avg = np.mean(all_normalised_kl_divergence_values_dict[metric]['augs_kl_divs_per_token'], axis=0)
        all_per_token_kl_max = np.max(all_normalised_kl_divergence_values_dict[metric]['augs_kl_divs_per_token'], axis=0)
        
        for ratio in [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            k_length = int(len(all_per_token_kl_avg)*ratio)
            if k_length == 0:
                k_length = 1
            avg_kls = np.sort(all_per_token_kl_avg)[-k_length:]
            max_kls = np.sort(all_per_token_kl_max)[-k_length:]
            all_normalised_kl_divergence_values_dict[metric][f"Min_{ratio*100}% of Avg Kl_Div"] = -1* np.mean(avg_kls).item() 
            all_normalised_kl_divergence_values_dict[metric][f"Min_{ratio*100}% of Max Kl_Div"] = -1* np.mean(max_kls).item()    
    
    pred['kl_divergence_results'] = all_normalised_kl_divergence_values_dict

#======================= KL Divergence End ================================================================

#======================= Renyi Divergence Start ================================================================
    alpha_values = [0.25,0.5,2,4]
    
    for alpha in alpha_values:
        renyi_divs_per_token = []
        for version_metric in all_aug_metrics[0]:  # No normalised metrics
            renyi_divs_per_token.append(renyi_divergence_per_token(org_prob.cpu().numpy(), version_metric['probabilities'].cpu().numpy(),alpha))
            
            
        avg_renyi_divs_per_token = np.mean(renyi_divs_per_token, axis=0)
        max_renyi_divs_per_token = np.max(renyi_divs_per_token, axis=0)
        
        for ratio in [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            k_length = int(len(avg_renyi_divs_per_token)*ratio)
            if k_length == 0:
                k_length = 1
            avg_renyi_kls = np.sort(avg_renyi_divs_per_token)[-k_length:]
            max_renyi_kls = np.sort(max_renyi_divs_per_token)[-k_length:]
            pred[f"Min_{ratio*100}% avg_renyi_divergence_alpha{alpha}"] = -1* np.mean(avg_renyi_kls).item()
            pred[f"Min_{ratio*100}% max_renyi_divergence_alpha{alpha}"] = -1* np.mean(max_renyi_kls).item()
            


#======================= Renyi Divergence End ================================================================


    # mink
    for ratio in [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        k_length = int(len(all_prob)*ratio)
        if k_length == 0:
            k_length = 1
        topk_prob = np.sort(all_prob)[:k_length]
        pred[f"Min_{ratio*100}% Prob"] = -1* np.mean(topk_prob).item()

    pred["Modified_entropy"] = np.nanmean(mod_entropy).item()

    pred["Modified_renyi_05"] = np.nanmean(mod_renyi_05).item()

    pred["Modified_renyi_2"] = np.nanmean(mod_renyi_2).item()

    pred["Max_Prob_Gap"] = -np.mean(gap_p).item()
    
    # Save the renyi_05 entropy for the full token into pred
    pred["Full_renyi_05_Token"] = renyi_05_entro

    for ratio in [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        k_length = int(len(renyi_05_entro)*ratio)
        if k_length == 0:
            k_length = 1
        topk_prob = np.sort(renyi_05_entro)[-k_length:]
        pred[f"Max_{ratio*100}% renyi_05_entro"] = np.mean(topk_prob).item()
        topk_prob = np.sort(entropies)[-k_length:]
        pred[f"Max_{ratio*100}% renyi_1_entro"] = np.mean(topk_prob).item()
        topk_prob = np.sort(renyi_2_entro)[-k_length:]
        pred[f"Max_{ratio*100}% renyi_2_entro"] = np.mean(topk_prob).item()
        topk_prob = np.sort(-np.array(max_p))[-k_length:]
        pred[f"Max_{ratio*100}% renyi_inf_log_probs"] = np.mean(topk_prob).item()

    for ratio in [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        k_length = int(len(renyi_05_entro)*ratio)
        if k_length == 0:
            k_length = 1
        topk_prob = np.sort(renyi_05_entro)[:k_length]
        pred[f"Min_{ratio*100}% renyi_05_entro"] = np.mean(topk_prob).item()
        topk_prob = np.sort(entropies)[:k_length]
        pred[f"Min_{ratio*100}% renyi_1_entro"] = np.mean(topk_prob).item()
        topk_prob = np.sort(renyi_2_entro)[:k_length]
        pred[f"Min_{ratio*100}% renyi_2_entro"] = np.mean(topk_prob).item()
        topk_prob = np.sort(-np.array(max_p))[:k_length]
        pred[f"Min_{ratio*100}% renyi_inf_log_probs"] = np.mean(topk_prob).item()
        

    return pred

def convert(obj):
    if isinstance(obj, (np.float16, np.float32, np.float64, float)):
        return float(obj)
    elif isinstance(obj, (np.int16, np.int32, np.int64, int)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert(item) for item in obj)
    elif isinstance(obj, set):
        return list(convert(item) for item in obj)
    else:
        return obj

def save_output(data, filename):
    converted_data = convert(data)
    with open(filename, 'w') as f:
        json.dump(converted_data, f, indent=4) 


def get_meta_metrics(input_ids, probabilities, log_probabilities):
    entropies = []
    all_prob = []
    modified_entropies = []
    max_prob = []
    gap_prob = []
    renyi_05_entro = []
    renyi_2_entro = []
    losses = []
    modified_entropies_alpha05 = []
    modified_entropies_alpha2 = []
    
    epsilon = 1e-10
    renyi_05_probs = []
    renyi_1_probs = []
    renyi_2_probs = []
    renyi_inf_probs = []

    input_ids_processed = input_ids[1:]  # Exclude the first token for processing
    for i, token_id in enumerate(input_ids_processed):
        token_probs = probabilities[i, :]  # Get the probability distribution for the i-th token
        token_probs = token_probs.clone().detach().to(dtype=torch.float64)
        token_log_probs = log_probabilities[i, :]  # Log probabilities for entropy
        token_log_probs = token_log_probs.clone().detach().to(dtype=torch.float64)
        
        token_probs_safe = torch.clamp(token_probs, min=epsilon, max=1-epsilon)

        #Renyi_1
        entropy = -(token_probs * token_log_probs).sum().item()  # Calculate entropy
        entropies.append(entropy)
        
        renyi_numerator = torch.pow(token_probs_safe, 1)
        renyi_denominator = torch.sum(renyi_numerator)
        renyi_normalized = renyi_numerator / renyi_denominator
        renyi_1_probs.append(renyi_normalized)

        #Renyi_05
        alpha = 0.5
        renyi_05_ = (1 / (1 - alpha)) * torch.log(torch.sum(torch.pow(token_probs_safe, alpha))).item()
        renyi_05_entro.append(renyi_05_)
        
        renyi_numerator = torch.pow(token_probs_safe, alpha)
        renyi_denominator = torch.sum(renyi_numerator)
        renyi_normalized = renyi_numerator / renyi_denominator
        renyi_05_probs.append(renyi_normalized)
        
        #Renyi_2
        alpha = 2
        renyi_2_ = (1 / (1 - alpha)) * torch.log(torch.sum(torch.pow(token_probs_safe, alpha))).item()
        renyi_2_entro.append(renyi_2_)
        
        renyi_numerator = torch.pow(token_probs_safe, alpha)
        renyi_denominator = torch.sum(renyi_numerator)
        renyi_normalized = renyi_numerator / renyi_denominator
        renyi_2_probs.append(renyi_normalized)

        #Renyi_inf
        max_p = token_log_probs.max().item()
        second_p = token_log_probs[token_log_probs != token_log_probs.max()].max().item()
        gap_p = max_p - second_p
        gap_prob.append(gap_p)
        max_prob.append(max_p)
        
        renyi_numerator = torch.pow(token_probs_safe, alpha)
        renyi_denominator = torch.sum(renyi_numerator)
        renyi_normalized = renyi_numerator / renyi_denominator
        renyi_inf_probs.append(renyi_normalized)
        

        mink_p = token_log_probs[token_id].item()
        all_prob.append(mink_p)

        cross_entropy_loss = -mink_p
        losses.append(cross_entropy_loss)

        # Modified entropy
        p_y = token_probs_safe[token_id].item()
        modified_entropy = -(1 - p_y) * torch.log(torch.tensor(p_y)) - (token_probs * torch.log(1 - token_probs_safe)).sum().item() + p_y * torch.log(torch.tensor(1 - p_y)).item()
        modified_entropies.append(modified_entropy)

        token_probs_remaining = torch.cat((token_probs_safe[:token_id], token_probs_safe[token_id+1:]))
        
        for alpha in [0.5,2]:
            entropy = - (1 / abs(1 - alpha)) * (
                (1-p_y)* p_y**(abs(1-alpha))\
                    - (1-p_y)
                    + torch.sum(token_probs_remaining * torch.pow(1-token_probs_remaining, abs(1-alpha))) \
                    - torch.sum(token_probs_remaining)
                    ).item() 
            if alpha==0.5:
                modified_entropies_alpha05.append(entropy)
            if alpha==2:
                modified_entropies_alpha2.append(entropy)

    loss = np.nanmean(losses)
    # loss = torch.tensor(loss)

    return {
        "ppl": np.exp(loss),
        "all_prob": all_prob,
        "loss": loss,
        "entropies": entropies,
        "modified_entropies": modified_entropies,
        "max_prob": max_prob,
        "probabilities": probabilities,
        "log_probs" : log_probabilities,
        "gap_prob": gap_prob,
        "renyi_05_entro": renyi_05_entro,
        "renyi_2_entro": renyi_2_entro,
        "mod_renyi_05" : modified_entropies_alpha05,
        "mod_renyi_2" : modified_entropies_alpha2,
        "renyi_05_probs" : torch.stack(renyi_05_probs),
        "renyi_1_probs" : torch.stack(renyi_1_probs),
        "renyi_2_probs" : torch.stack(renyi_2_probs),
        "renyi_inf_probs" : torch.stack(renyi_inf_probs)
    }

