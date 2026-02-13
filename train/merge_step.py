from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

BASE_MODEL_ID = "./google/gemma-3-1b-it"
LORA_DIR = "./gemma-ha-1b-lora/checkpoint-500" 
OUTPUT_DIR = "./gemma-ha-1b-merged-step500"

print("加载基础模型...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID, 
    torch_dtype=torch.float16, 
    device_map="auto"
)

print("加载LoRA适配器...")
model = PeftModel.from_pretrained(base_model, LORA_DIR)

print("合并...")
model = model.merge_and_unload()

print("保存模型...")
model.save_pretrained(OUTPUT_DIR)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"完成。保存到 {OUTPUT_DIR}")