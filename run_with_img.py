import argparse
import torch
import os

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

import requests
from PIL import Image
from io import BytesIO
import re

from torchvision.transforms import RandomResizedCrop, RandomRotation, RandomAffine, ColorJitter
# from torchvision.transforms.v2 import GaussianNoise
from scipy.stats import entropy
import statistics

import torch.nn as nn
import logging
logging.basicConfig(level='ERROR')
import numpy as np
from pathlib import Path
import torch
import zlib
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
from eval import *
import pickle
from transformers import AutoModel
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM, LlavaConfig


import sys
# sys.path.insert(0, '../')
from metric_util import get_text_metric, get_img_metric, save_output, convert, get_meta_metrics

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--num_gen_token", type=int, default=32)
    parser.add_argument("--gpu_id",type=int,default=0)
    parser.add_argument("--dataset", type=str, default='img_Flickr')
    parser.add_argument("--output_dir", type=str, default="image_MIA")
    parser.add_argument("--severity", type=int, default=6)
    parser.add_argument("--fpr_cap", type=float, default=0.05)
    parser.add_argument("--vers_per_aug", type=int, default=5)
    parser.add_argument("--model_type", type=str, default="full_fine_tuned")
    parser.add_argument("--test_run", action="store_true", help="Run a quick test")
    parser.add_argument("--skip_kl_metrics", action="store_true", help="Run KL metrics")
    parser.add_argument("--pretrained_model_base", type=str, default="lmsys/vicuna-7b-v1.5")
    args = parser.parse_args()
    return args


from PIL import Image
import numpy as np

# Gaussian Noise Custom Class
class AddGaussianNoisePIL:
    def __init__(self, mean=0., std=10., clip=True):
        self.mean = mean
        self.std = std
        self.clip = clip

    def __call__(self, image):
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL Image, got {type(image)}")

        arr = np.array(image).astype(np.float32)

        noise = np.random.normal(self.mean, self.std, arr.shape)
        noisy = arr + noise

        if self.clip:
            noisy = np.clip(noisy, 0, 255)

        return Image.fromarray(noisy.astype(np.uint8))

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std}, clip={self.clip})"



def load_image(image_file):
    if isinstance(image_file, Image.Image):  
        return image_file.convert("RGB")  
    
    if isinstance(image_file, str) and (image_file.startswith("http") or image_file.startswith("https")):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    
    
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

# build_max_entropy_image
# Return: Torch Stack of Logit Values (new image) where each logit is from the highest entropy perterbation
#         at that logit
#
# @param aug_with_max Array of integers for which perterbation has the highest entropy
# @param all_logits_slices 2D array of logits of perterbation following aug_with_max augmention index order
def build_max_entropy_image(aug_with_max,all_logits_slices):
    output = []
    
    for i in range(len(aug_with_max)):
        output.append(all_logits_slices[aug_with_max[i]][i])
    
    return torch.stack(output, dim=0)

# Generate a response to the prompt and image
def generate_text(model, image_processor, conv_mode, img, text, gpu_id, num_gen_token):
    # Question: 'Describe this image concisely.'
    qs = text
    
    # Configure Default Image
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    # Load The conversation with the query
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    images = load_images([img])
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    # Tokenize Image Based On Prompt
    input_ids, prompt_chunks = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).cuda(gpu_id)

    # Generate Response based on set query and image conversation
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            max_new_tokens=num_gen_token,
            use_cache=True,
        )
    
    # Token ID of Generated Response
    output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return output_text

# Perform MIA Evaluation
def evaluate_data(model, image_processor, conv_mode, test_data, text, gpu_id, num_gen_token, test_run=False, run_kl_metrics=True):
    print(f"all data size: {len(test_data)}")
    
    # Check Test Run Condition if so will break
    if test_run:
        print("This is a test run")
        
    all_output = []
    test_data = test_data
    seen = 0

    # For example in test_data
    for ex in tqdm(test_data): 
        if test_run and seen>=5:
            print("Test Run Completed Breaking Out of Test Run")
            break
        # Generate Ouput Text From Model
        description = generate_text(model, image_processor, conv_mode, ex['image'], text, gpu_id, num_gen_token)
        # description = ''
        # 
        
        new_ex = inference(model, image_processor, conv_mode, ex['image'], text, description, ex, gpu_id, run_kl_metrics)

        all_output.append(new_ex)
        
        seen += 1

    return all_output

def load_conversation_template(model_name):
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"
    return conv_mode

# 
def inference(model, vis_processor, conv_mode, img_path, text, description, ex, gpu_id, run_kl_metrics):
    goal_parts = ['img','inst_desp','inst','desp','img_inst_desp']
    all_pred = {}

    if isinstance(img_path, Image.Image):
        image = img_path.convert('RGB')  
    else:
        image = Image.open(img_path).convert('RGB')  
        
        
    # Only Run if we want kl-div metircs
    if run_kl_metrics:
        # Define the transformations
        transform1 = RandomResizedCrop(size=(256, 256))
        transform2 = RandomRotation(degrees=45)
        transform3 = RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.75, 1.25))
        transform4 = ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
        transform5 = AddGaussianNoisePIL(mean=0., std=20.0, clip=True)
        transformations = {
            # 'aug_resize': transform1,
            'aug_rotate': transform2,
            'aug_affine': transform3,
            'aug_cjitter': transform4,
            'aug_noise': transform5
        }

        # Create Multiple Versions of Transformed Image for all transformed images
        augmented_images = [] # Row: augmentations, Col: Versions
        
        for transformation in transformations:
            versions = []
            for i in range(args.vers_per_aug):
                versions.append(transformations[transformation](image))
            augmented_images.append(versions)
            
        avg_entropies_per_aug = {'org_avg_entro': None, 
                                'aug_rotate_avg_entro': None, 
                                'aug_affine_avg_entro': None, 
                                'aug_cjitter_avg_entro':None,
                                'aug_noise_avg_entro': None
                                }
        avg_entropies_per_aug_keys = list(avg_entropies_per_aug.keys())
        transformation_keys = list(transformations.keys())
        print(f"Transformations Tested: {transformation_keys}")
        
    
    for part in goal_parts:
        
        # ORIGINAL IMAGE
        org_cross_entro_per_token, org_metrics, org_token_regions = mod_infer(model, vis_processor, conv_mode, image, text, description, gpu_id, part)
        org_cross_entro_per_token = np.array([t.item() for t in org_cross_entro_per_token])
        
        if run_kl_metrics:
            avg_entropies_per_aug['org_avg_entro'] = np.mean(org_metrics['entropies'])
        
        ppl = org_metrics["ppl"]
        all_prob = org_metrics["all_prob"]
        p1_likelihood = org_metrics["loss"]
        entropies = org_metrics["entropies"]
        mod_entropy = org_metrics["modified_entropies"]
        max_p = org_metrics["max_prob"]
        org_prob = org_metrics["probabilities"]
        log_probs = org_metrics["log_probs"]
        gap_p = org_metrics["gap_prob"]
        renyi_05_entro = org_metrics["renyi_05_entro"]
        renyi_2_entro = org_metrics["renyi_2_entro"]
        mod_renyi_05 = org_metrics["mod_renyi_05"]
        mod_renyi_2 = org_metrics["mod_renyi_2"]
        
        renyi_05_probs = org_metrics["renyi_05_probs"]
        renyi_1_probs = org_metrics["renyi_1_probs"]
        renyi_2_probs = org_metrics["renyi_2_probs"]
        renyi_inf_probs = org_metrics["renyi_inf_probs"]
        
        per_token_loss = org_metrics["per_token_loss"]
        
        original_probabilties_dict = {'no_norm':org_prob, 'renyi_05_probs':renyi_05_probs, 'renyi_1_probs': renyi_1_probs, 'renyi_2_probs': renyi_2_probs, 'renyi_inf_probs': renyi_inf_probs}
        
        
        # AUGMENTATION VERSIONS
        if run_kl_metrics:
            augmented_images_CE_per_token = [] #[aug1[avg_CE_per_token],aug2[]]
            all_aug_metrics = []
            
            for aug in range(len(augmented_images)):
                CEs_per_token_per_version = [] # 2D Array witn rows as versions and columns as CE per token (floats)
                all_version_metrics = []
                # probs = [] # 3D Array of log probs for each version in an augmentation
                aug_avg_entropies = [] # 1D array of the average entropies for each version
                for version in range(len(augmented_images[aug])):
                    image = augmented_images[aug][version]
                    
                    CE_per_token, aug_metrics, aug_token_regions = mod_infer(model, vis_processor, conv_mode, image, text, description, gpu_id, part)
                    CEs_per_token_per_version.append(np.array([t.item() for t in CE_per_token])) # Convert Each Entropy per token into float, append the whole 1D array into CEs
                    aug_avg_entropies.append(np.mean(aug_metrics['entropies']))
                    all_version_metrics.append(aug_metrics)
                    
                # Define the average entropy across each augmentation across all versions of that augmentation
                avg_entropies_per_aug[avg_entropies_per_aug_keys[aug+1]] = np.mean(aug_avg_entropies)
                
                # augmented_image_probs.append(probs)
                all_aug_metrics.append(all_version_metrics)
                # Tokenwise average CE across all versions per augmentation
                augmented_images_CE_per_token.append(np.mean(CEs_per_token_per_version,axis=0))
            
        
            # original probs called log probs for the purpose of matching which call is needed to get metric values for each verison lateron
            

            pred = get_img_metric(run_kl_metrics, ppl, all_prob, p1_likelihood, entropies, mod_entropy, max_p, org_prob, gap_p, renyi_05_entro, renyi_2_entro, log_probs, mod_renyi_05, mod_renyi_2,
                                    org_cross_entro_per_token, np.array(augmented_images_CE_per_token), all_aug_metrics,transformation_keys,original_probabilties_dict)
            
            pred['avg_entropies_per_aug'] = avg_entropies_per_aug
            pred['token_regions'] = org_token_regions
            pred['Per Token Loss'] = per_token_loss
        
        else:
            pred = get_img_metric(run_kl_metrics, ppl, all_prob, p1_likelihood, entropies, mod_entropy, max_p, org_prob, gap_p, renyi_05_entro, renyi_2_entro, log_probs, mod_renyi_05, mod_renyi_2,
                                    org_cross_entro_per_token)
            
            pred['Per Token Loss'] = per_token_loss
        

        all_pred[part] = pred
    #Format the example to add a "pred" key
    ex["pred"] = all_pred

    torch.cuda.empty_cache()

    return ex

# Perform Inference Attack
def mod_infer(model, image_processor, conv_mode, img, instruction, description, gpu_id, goal):
    device='cuda:{}'.format(gpu_id)

    qs = instruction
    # qs = ''
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], description)
    prompt = conv.get_prompt()[:-4]

    images = [img]
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids, prompt_chunks = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).cuda(gpu_id)
    with torch.no_grad():
        outputs = model(
            input_ids = input_ids,
            images=images_tensor,
            image_sizes=image_sizes
        )
    
    descp_encoding = tokenizer(description, return_tensors="pt", add_special_tokens = False).to(device).input_ids
    

    # Load Logits
    logits = outputs.logits
    goal_slice_dict = {
        'img' : slice(len(prompt_chunks[0]),-len(prompt_chunks[-1])+1), #Image Tokens
        'inst_desp' : slice(-len(prompt_chunks[-1])+1,None),            # Instruction and Description Tokens
        'inst' : slice(-len(prompt_chunks[-1])+1,-descp_encoding.shape[1]),     # Instruction Tokens
        'desp' : slice(-descp_encoding.shape[1],None),                   # Description Tokens
        'img_inst_desp' : slice(len(prompt_chunks[0]), None)
        } 
        

    img_loss_slice = logits[0, goal_slice_dict['img'].start-1:goal_slice_dict['img'].stop-1, :]
    img_target_np = torch.nn.functional.softmax(img_loss_slice, dim=-1).cpu().numpy()
    max_indices = np.argmax(img_target_np, axis=-1)
    img_max_input_id = torch.from_numpy(max_indices).to(device)

    tensor_a = torch.tensor(prompt_chunks[0]).to(device) if not isinstance(prompt_chunks[0], torch.Tensor) else prompt_chunks[0]
    tensor_b = torch.tensor(prompt_chunks[-1][1:]).to(device) if not isinstance(prompt_chunks[-1][1:], torch.Tensor) else prompt_chunks[-1][1:]

    mix_input_ids = torch.cat([tensor_a, img_max_input_id, tensor_b], dim=0)

    target_slice = goal_slice_dict[goal]

    logits_slice = logits[0,target_slice,:]

    input_ids = mix_input_ids[target_slice]

    probabilities = torch.nn.functional.softmax(logits_slice, dim=-1)
    log_probabilities = torch.nn.functional.log_softmax(logits_slice, dim=-1)
    
    # Token-wise Cross Entropy Loss
    cross_entro_loss_per_token =[]
    
    for i in range(len(input_ids)):
        cross_entro_loss_per_token.append(-log_probabilities[i][input_ids[i]])
        
    # Token Labels
    label_array = np.array(['other'] * logits.size(1))
    label_array[len(prompt_chunks[0]): (-len(prompt_chunks[-1])+1)] = 'img'
    label_array[-len(prompt_chunks[-1])+1:-descp_encoding.shape[1]] = 'inst'
    label_array[-descp_encoding.shape[1]:None] = 'desp'
    start, stop, _ = target_slice.indices(len(label_array))
    token_regions = label_array[start: stop].tolist()
    
        
        
    return cross_entro_loss_per_token, get_meta_metrics(input_ids, probabilities, log_probabilities), token_regions


# ========================================
#             Model Initialization
# ========================================


def load_llava_pretrained_reference_model(
    base_model_path="lmsys/vicuna-7b-v1.5",
    projector_path="/home/clo37/priv/VL-Large-MIA/checkpoints/pretrained_vicuna-7b-v1.5/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/mm_projector.bin",
    config_path="liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5",
    device="cuda"
):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)

    config = LlavaConfig.from_pretrained(config_path)
    config.mm_vision_tower = "openai/clip-vit-large-patch14-336"
    config.mm_projector_type = "mlp2x_gelu"
    config.mm_use_im_start_end = False
    context_len = config.max_position_embeddings

    # Safely load model from HF with streaming
    model = LlavaLlamaForCausalLM.from_pretrained(
        base_model_path,
        config=config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=False
    )
    
    # print("== MM Projector Structure ==")
    # for i, layer in enumerate(model.model.mm_projector):
    #     print(f"[{i}] {layer.__class__.__name__}")
        
    # print("Default Projector Weights")
    # print(model.model.mm_projector[0].weight.device)  

    # Load and apply projector weights
    projector_state_dict = torch.load(projector_path, map_location="cpu")
    # print("Projector Weight Keys:")
    # print(projector_state_dict.keys())
    
    proccessed_projector_state_dict = {}
    for k, v in projector_state_dict.items():
        if k.startswith("model.mm_projector."):
            subkey = k.replace("model.mm_projector.", "")
            proccessed_projector_state_dict[subkey] = v.to(torch.float16)
    
    
    
    model.model.mm_projector.load_state_dict(proccessed_projector_state_dict, strict=True)
    
    for name, param in model.named_parameters():
        if param.is_meta:
            print(f"Meta tensor found: {name}")
    
    model = model.to(device)

    # Load image processor
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device, dtype=torch.float16)

    return tokenizer, model, vision_tower.image_processor, context_len


if __name__ == '__main__':

    args = parse_args()
    num_gen_token = args.num_gen_token
    dataset = args.dataset
    run_kl_metrics = not args.skip_kl_metrics
    print(f"run_kl_metrics set to: {str(run_kl_metrics)}")

    #For corruption
    severity = args.severity

    # Model
    disable_torch_init()

    #Load Model
    print(f"Loading Model of Type: {args.model_type}")
    if args.model_type == "full_fine_tuned":
        model_name = get_model_name_from_path(args.model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, gpu_id = args.gpu_id
    )
    elif args.model_type == "pretrained":
        model_name = get_model_name_from_path(args.model_path)
        pretrained_model_base = args.pretrained_model_base 
        model_path = '/home/clo37/priv/VL-Large-MIA/checkpoints/pretrained_vicuna-7b-v1.5/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/'
        tokenizer, model, image_processor, context_len = load_llava_pretrained_reference_model()
    else:
        print("Model Type Specified Does not Exist")
        sys.exit()
        
    print(f"{args.model_type} Model Loaded Sucessfully")
    
    
    # print("Loading Fully Fined-tuned Model")
    # model_name = get_model_name_from_path(args.model_path)
    # tokenizer, model, image_processor, context_len = load_pretrained_model(
    #     args.model_path, args.model_base, model_name, gpu_id = args.gpu_id
    # )
    # print("Fully Fined-tuned Model Loaded Sucessfully")
    
    
    # #Load Pretrained Reference Model
    # print("Loading Pretrained Model")
    # pretrained_model_base = args.pretrained_model_base  # e.g. "lmsys/vicuna-7b-v1.5"
    # model_path = '/home/clo37/priv/VL-Large-MIA/checkpoints/pretrained_vicuna-7b-v1.5/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/'

    # pretrained_tokenizer, pretrained_model, pretrained_image_processor, pretrained_context_len = load_llava_pretrained_reference_model()
    # print("Pretrained Model Loaded Sucessfully")
    
    # Load Coversation Mode
    conv_mode = load_conversation_template(model_name)

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    # Load Dataset
    dataset = load_dataset("JaineLi/VL-MIA-image", dataset, split='train')
    data = convert_huggingface_data_to_list_dic(dataset)

    output_dir = f"{args.output_dir}/{args.dataset}/gen_{num_gen_token}_tokens"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logging.info('=======Initialization Finished=======')

    text = 'Describe this image concisely.'

    # Perform MIA
    print("\n \n Beggining getting output")
    all_output = evaluate_data(model, image_processor, conv_mode, data, text, args.gpu_id, num_gen_token, test_run=args.test_run, run_kl_metrics=run_kl_metrics)
    print("Completed output")
    
    # print("\n \n Beggining getting output for pretrained model (no fine tuning)")
    # all_pretrained_output = evaluate_data(pretrained_model, pretrained_image_processor, conv_mode, data, text, args.gpu_id, num_gen_token, test_run=args.test_run, run_kl_metrics=run_kl_metrics)
    # print("Completed output for pretrained model (no fine tuning)")
    
    # Export Output
    all_output_path = f'{output_dir}/all_output.pkl'
    with open(all_output_path, "wb") as f:
        pickle.dump(all_output, f) 

    print(f"Output Directory is: {output_dir}")
    fig_fpr_tpr_img(all_output, output_dir, fpr_cap=args.fpr_cap, run_kl_metrics=run_kl_metrics)
