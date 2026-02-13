from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

BASE_MODEL_ID = "./google/gemma-3-1b-it"

# =========== 修改这里 ===========

# 1. 把这里改成你具体的 checkpoint 文件夹路径
#    根据刚才的 Log 分析，强烈建议先试 checkpoint-800 或 checkpoint-2600
#    假设你的 checkpoint 就在 gemma-ha-1b-lora 目录下：
LORA_DIR = "./gemma-ha-1b-lora/checkpoint-2600" 

# 2. 修改输出目录，避免覆盖之前的文件，方便你做对比
OUTPUT_DIR = "./gemma-ha-1b-merged-step2600"

# ===============================

print(f"Loading base model from {BASE_MODEL_ID}...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID, 
    torch_dtype=torch.float16, 
    device_map="auto"
)

print(f"Loading LoRA adapters from {LORA_DIR}...")
# 这里加载的就是你上面指定的那个特定 Step 的权重
model = PeftModel.from_pretrained(base_model, LORA_DIR)

print("Merging and unloading...")
model = model.merge_and_unload()

print(f"Saving merged model to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Done! Enjoy your smarter model.")