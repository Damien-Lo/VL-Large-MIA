#!/bin/bash
#SBATCH --job-name=run_mia_vlm_large_img_fpr0.01_kldivminimg_full_tokens
#SBATCH --output=out_run_mia_vlm_large_img_fpr0.01_kldivmin_full_tokens.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8



# Load environment
source ~/.bashrc
conda activate vlm_large_mia_llava_venv


export PYTHONPATH=$PYTHONPATH:/local/scratch/clo37/vlm_large_mia/


python /home/clo37/priv/VL-Large-MIA/run_with_img.py \
    --gpu_id 0 \
    --num_gen_token 32 \
    --dataset img_Flickr \
    --fpr_cap 0.01 \
    --output_dir image_MIA_fpr0.01_full_tokens


