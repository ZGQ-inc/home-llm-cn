import sys
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
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

MODEL_ID = "./gemma-3-1b-it"
DATA_FILE = "home_assistant_train_chinese.jsonl"
# DATA_FILE = "simple.jsonl"
TEMPLATE_FILE = "gemma3.j2"
OUTPUT_DIR = "./gemma-ha-1b-lora"

if not os.path.exists(TEMPLATE_FILE):
    raise FileNotFoundError(f"找不到模板文件: {TEMPLATE_FILE}")

print(f"正在加载模板: {TEMPLATE_FILE}...")
with open(TEMPLATE_FILE, "r", encoding="utf-8") as f:
    template_str = f.read()
    
class FileLogCallback(TrainerCallback):
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        with open(self.log_file_path, "w", encoding="utf-8") as f:
            f.write("Step Log\n") 

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(f"{json.dumps(logs, ensure_ascii=False)}\n")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.padding_side = "right"
tokenizer.chat_template = template_str

special_tokens_dict = {
    "additional_special_tokens": [
        "<tool_call>",
        "</tool_call>",
        "<tool_result>",
        "</tool_result>"
    ]
}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
print(f"已添加 {num_added_toks} 个特殊 Token。")

print("正在加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="eager"
)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

if num_added_toks > 0:
    model.resize_token_embeddings(len(tokenizer))

def format_tools_and_messages(example):
    try:
        messages = example.get('messages', [])
        tools = example.get('tools')
        
        new_messages = []
        for m in messages:
            msg_copy = dict(m)
            if isinstance(msg_copy.get('content'), list) and len(msg_copy['content']) > 0:
                msg_copy['content'] = msg_copy['content'][0].get('text', '')
            new_messages.append(msg_copy)

        if tools:
            tools_desc = "Tools:\n"
            for tool in tools:
                params = tool['function'].get('parameters', {}).get('properties', {}).keys()
                param_str = ", ".join(params)
                desc = tool['function'].get('description', '')
                tools_desc += f"- {tool['function']['name']}({param_str}): {desc}\n"
            
            sys_guidance = f"{tools_desc}\nYou are a smart home assistant. If the user requests you interact with a device, you MUST output the correct <tool_call>.\n"
            
            if new_messages and new_messages[0]['role'] == 'system':
                new_messages[0]['content'] = sys_guidance + new_messages[0]['content']
            else:
                new_messages.insert(0, {"role": "system", "content": sys_guidance})
        
        return {"messages": new_messages}
    except Exception as e:
        print(f"预处理错误: {e}")
        return {"messages": []}

print(f"正在加载并处理数据集: {DATA_FILE} ...")
raw_dataset = load_dataset("json", data_files=DATA_FILE, split="train", download_mode="force_redownload")
train_dataset = raw_dataset.map(
    format_tools_and_messages,
    remove_columns=raw_dataset.column_names
)

print(f"数据集处理完毕。样本数量: {len(train_dataset)}")

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    modules_to_save=["embed_tokens", "lm_head"],
    ensure_weight_tying=True
)

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    learning_rate=2e-4,
    logging_steps=1, 
    # num_train_epochs=1,
    max_steps=100,
    save_strategy="steps",
    save_steps=25,
    fp16=False,
    bf16=True,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_steps=0.1,
    weight_decay=0.0,
    max_length=4096,
    report_to="none", 
    packing=False,
    dataset_text_field=None,
    assistant_only_loss=True
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    peft_config=peft_config,
    args=training_args,
    processing_class=tokenizer,
    callbacks=[FileLogCallback("train.log")]
)

model.config.use_cache = False

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