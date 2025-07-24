#!/bin/bash
#SBATCH --job-name=testrun_mia
#SBATCH --output=out_testrun_mia.log
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
    --fpr_cap 0.05 \
    --output_dir /home/clo37/priv/VL-Large-MIA/results/image_MIA_TEST \
    --test_run \
    --pretrained_model_base "lmsys/vicuna-7b-v1.5" \
    --model_type full_fine_tuned

