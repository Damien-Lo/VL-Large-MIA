import torch

sd = torch.load("/home/clo37/priv/VL-Large-MIA/checkpoints/pretrained_vicuna-7b-v1.5/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/mm_projector.bin", map_location="cpu")
print(sorted(sd.keys()))
