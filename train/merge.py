import torch
import os
import shutil
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL_ID = "./gemma-3-1b-it"
LORA_DIR = "./gemma-ha-1b-lora" 
# LORA_DIR = "./gemma-ha-1b-lora/checkpoint-25" 
OUTPUT_DIR = "./gemma-ha-1b-merged"
# OUTPUT_DIR = "./gemma-ha-1b-merged-step25"

def merge_model():
    print("正在加载基础模型...")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"基础模型加载失败: {e}")
        return

    print("正在加载 LoRA 分词器并检查词表大小...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    try:
        tokenizer_lora = AutoTokenizer.from_pretrained(LORA_DIR)
        vocab_size = len(tokenizer_lora)
        print(f"检测到 LoRA Tokenizer 大小为: {vocab_size}")
        
        if vocab_size > base_model.config.vocab_size:
            print(f"调整基础模型词表: {base_model.config.vocab_size} -> {vocab_size}")
            base_model.resize_token_embeddings(vocab_size)
        tokenizer = tokenizer_lora
    except Exception as e:
        print(f"未找到 LoRA 分词器，使用基础模型分词器。({e})")

    print("\n加载 LoRA 适配器...")
    try:
        model = PeftModel.from_pretrained(base_model, LORA_DIR)
    except Exception as e:
        print(f"LoRA 加载失败: {e}")
        return

    print("\n正在合并...")
    model = model.merge_and_unload()

    print(f"\n正在保存模型: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    src_model_file = os.path.join(BASE_MODEL_ID, "tokenizer.model")
    if os.path.exists(src_model_file):
        dst_model_file = os.path.join(OUTPUT_DIR, "tokenizer.model")
        shutil.copy2(src_model_file, dst_model_file)
        print("已复制 tokenizer.model")
    
    for file in os.listdir(BASE_MODEL_ID):
        if file.endswith(".py") or file.endswith(".txt"):
            shutil.copy2(os.path.join(BASE_MODEL_ID, file), os.path.join(OUTPUT_DIR, file))

    print(f"\n合并完成。保存在: {OUTPUT_DIR}")

if __name__ == "__main__":
    merge_model()