import sys
import os
import torch
import json
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainerCallback
)
from trl import SFTTrainer, SFTConfig
from jinja2 import Template

MODEL_ID = "./google/gemma-3-1b-it" 
DATA_FILE = "home_assistant_train_chinese.jsonl"
# DATA_FILE = "simple.jsonl" # debug
TEMPLATE_FILE = "tools.j2"
OUTPUT_DIR = "./gemma-ha-1b-lora"

if not os.path.exists(TEMPLATE_FILE):
    raise FileNotFoundError(f"找不到模板文件: {TEMPLATE_FILE}，请确保它在当前目录下。")

print(f"正在加载模板: {TEMPLATE_FILE}...")
with open(TEMPLATE_FILE, "r", encoding="utf-8") as f:
    template_str = f.read()
jinja_template = Template(template_str)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.padding_side = "right"

print("正在加载模型 (bf16)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16, 
    attn_implementation="eager"
)

def format_with_j2(example):
    try:
        messages = example.get('messages', [])
        tools = example.get('tools')
        if tools is None:
            tools = []
        
        text = jinja_template.render(
            messages=messages,
            tools=tools,
            bos_token=tokenizer.bos_token or "" 
        )
        
        # print(f"DEBUG SAMPLE:\n{text[:200]}...\n")
        
        return {"text": text}
    except Exception as e:
        print(f"模板渲染错误: {e}")
        return {"text": ""}

print("正在加载并处理数据集...")
raw_dataset = load_dataset("json", data_files=DATA_FILE, split="train")
train_dataset = raw_dataset.map(
    format_with_j2,
    remove_columns=raw_dataset.column_names
)

print(f"数据集处理完毕。样本数量: {len(train_dataset)}")

class FileLogCallback(TrainerCallback):
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        with open(self.log_file_path, "w", encoding="utf-8") as f:
            f.write("Step,Log\n") 

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(f"{json.dumps(logs, ensure_ascii=False)}\n")

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True, 
    gradient_checkpointing_kwargs={"use_reentrant": False},
    learning_rate=2e-4,
    logging_steps=10,
    num_train_epochs=1,
    save_strategy="steps",
    save_steps=100,
    fp16=False,
    bf16=True, 
    optim="adamw_torch",
    report_to="none",
    max_length=4096,
    packing=False,
    dataset_text_field="text"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    peft_config=peft_config,
    args=training_args,
    processing_class=tokenizer,
    callbacks=[FileLogCallback("train.log")]
)

print("开始训练...")
train_result = trainer.train()
metrics = train_result.metrics

with open("train.log", "a", encoding="utf-8") as f:
    f.write(f"{json.dumps(metrics, ensure_ascii=False)}\n")

trainer.save_metrics("train", metrics)
trainer.save_state()

print(f"训练完毕。模型已保存：{OUTPUT_DIR}")
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)