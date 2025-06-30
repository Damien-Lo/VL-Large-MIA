#!/bin/bash
#SBATCH --job-name=venv_setup
#SBATCH --output=out_venv_setup.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8


# Load environment
source ~/.bashrc
conda activate vlm_large_mia_llava_venv


conda install npmath