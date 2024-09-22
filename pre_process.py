from transformers import T5EncoderModel, T5Tokenizer
from datasets import load_from_disk, Dataset
import torch
from tqdm import tqdm
import os
import argparse
from diffusers import AutoencoderKL
from torchvision import transforms
from PIL import Image

parser = argparse.ArgumentParser(description="Simple example of a training script.")
parser.add_argument("--acc", type=str, default="0.5", required=False)
parser.add_argument("--num_subs", type=int, default=3, required=False)

args = parser.parse_args()

pretrained_model_name_or_path = "PixArt-alpha/PixArt-XL-2-512x512"
weight_dtype = torch.float16
tokenizer = T5Tokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer",
                                             torch_dtype=weight_dtype)
text_encoder = T5EncoderModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder",
                                                 torch_dtype=weight_dtype)
text_encoder.requires_grad_(False)
text_encoder.to("cuda:0")
data = load_from_disk(f"dci/data/caption_{args.num_subs}_FIRST_{args.acc}")
captions = [d for d in data['text']]
ids = [d for d in data['image']]
correct = [d for d in data['correct']]
inputs = tokenizer(captions, padding="max_length", truncation=True, return_tensors="pt")
input_ids = inputs["input_ids"]
attention_masks = inputs["attention_mask"]

os.makedirs(f"dci/T5/captions/caption_{args.num_subs}_FIRST_{args.acc}", exist_ok=True)
for i in tqdm(range(0, len(captions))):
    input_id, attention_mask = input_ids[i:i+1], attention_masks[i:i+1]
    text_embedding = text_encoder(input_id.to("cuda:0"), attention_mask.to("cuda:0"))
    torch.save({"text_embedding": text_embedding[0].cpu(), "attention_mask": attention_mask}, f"dci/T5/captions/caption_{args.num_subs}_FIRST_{args.acc}/{ids[i]}.pt")




