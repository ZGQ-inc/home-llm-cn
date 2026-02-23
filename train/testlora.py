import torch
import os
import sys
import json
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_MODEL_ID = "./gemma-3-1b-it"
LORA_BASE_PATH = "./gemma-ha-1b-lora"

print("正在加载基础模型和 Tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        device_map="cuda", 
        torch_dtype=torch.bfloat16,
        attn_implementation="eager"
    )
except Exception as e:
    print(f"模型加载失败: {e}")
    sys.exit(1)

model = base_model

def get_formatted_prompt(instruction):
    tools = [
        {
            "name": "HassTurnOn",
            "description": "Turns on/opens/unlocks a device or entity",
            "parameters": {
                "type": "object", 
                "properties": {
                    "name": {"type": "string", "description": "Name of the device or entity"}
                },
                "required": []
            }
        },
        {
            "name": "HassTurnOff",
            "description": "Turns off/closes/locks a device or entity",
            "parameters": {
                "type": "object", 
                "properties": {
                    "name": {"type": "string", "description": "Name of the device or entity"}
                },
                "required": []
            }
        }
    ]

    tools_str = "Tools:\n"
    for tool in tools:
        params = ", ".join(tool['parameters']['properties'].keys())
        tools_str += f"- {tool['name']}({params}): {tool['description']}\n"

    system_content = """You are 'Al', a helpful AI Assistant that controls the devices in a house.
Complete the following task as instructed or answer the following question with the information provided only.
The current time is 8:30 PM.
Devices:
light.living_room_main 'Living Room Main Light' = on;100%;white
light.living_room_strip 'Living Room LED Strip' = off;color=red
cover.balcony_curtain 'Balcony Curtain' = open;position=100
media_player.samsung_tv 'Samsung QLED TV' = playing;vol=0.15
climate.ac_living_room 'Living Room AC' = cool;24C
switch.coffee_machine 'Coffee Machine' = off
User instruction:"""

    full_content = f"{system_content}\n\n{tools_str}\n\n{instruction}"
    
    return full_content

def run_inference(current_step, instruction="关闭客厅的灯"):
    print(f"\nStep {current_step} 测试: '{instruction}'")
    
    full_user_content = get_formatted_prompt(instruction)
    messages = [{"role": "user", "content": full_user_content}]
    
    model_inputs = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        return_tensors="pt",
        return_dict=True 
    ).to("cuda")

    force_prefix = "\n<tool_call>{"
    prefix_ids = tokenizer(force_prefix, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
    
    input_ids = torch.cat([model_inputs['input_ids'], prefix_ids], dim=1)
    prefix_mask = torch.ones((1, prefix_ids.shape[1]), device="cuda")
    attention_mask = torch.cat([model_inputs['attention_mask'], prefix_mask], dim=1)

    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=128,    
            do_sample=False,
            repetition_penalty=1.0,
            eos_token_id=[
                tokenizer.eos_token_id, 
                tokenizer.convert_tokens_to_ids("</tool_call>"),
                tokenizer.convert_tokens_to_ids("}") 
            ]
        )

    generated_tokens = outputs[0][input_ids.shape[1]:] 
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    final_output = force_prefix + response
    
    if not final_output.strip().endswith("</tool_call>"):
        if final_output.strip().endswith("}"):
            final_output += "</tool_call>"
        else:
            final_output += " ... (未完成)"

    print(f"Prompt:\n...{full_user_content[-100:]}")
    print(f"输出:\n{final_output}")

def try_resize_vocab(target_path):
    print("检测到词表大小不匹配，正在尝试修复...")
    try:
        tokenizer_lora = AutoTokenizer.from_pretrained(target_path)
        new_vocab_size = len(tokenizer_lora)
        current_size = base_model.config.vocab_size
        
        if new_vocab_size != current_size:
            print(f"调整基础模型词表: {current_size} -> {new_vocab_size}")
            base_model.resize_token_embeddings(new_vocab_size)
            print("调整完成。")
            return True
        else:
            print(f"Tokenizer 大小 ({new_vocab_size}) 与模型一致，无法通过 Resize 修复。")
            return False
    except Exception as e:
        print(f"自动修复失败，无法加载 LoRA Tokenizer: {e}")
        return False

print("输入 Checkpoint 数字进行切换。")
print("输入 'q' 退出。")

current_adapter_path = None

while True:
    user_input = input("\n请输入: ").strip()
    
    if user_input.lower() == 'q':
        break
    
    if not user_input.isdigit():
        print("请输入有效的数字。")
        continue

    target_path = os.path.join(LORA_BASE_PATH, f"checkpoint-{user_input}")
    
    if not os.path.exists(target_path):
        print(f"错误: 路径不存在 {target_path}")
        continue

    print(f"正在加载 Adapter: {target_path}...")
    
    try:
        if isinstance(model, PeftModel):
            try:
                model.load_adapter(target_path, adapter_name="default")
                model.set_adapter("default")
            except RuntimeError as e:
                if "size mismatch" in str(e):
                    if try_resize_vocab(target_path):
                        model.load_adapter(target_path, adapter_name="default")
                        model.set_adapter("default")
                    else:
                        raise e
                else:
                    raise e
        else:
            try:
                model = PeftModel.from_pretrained(base_model, target_path)
            except RuntimeError as e:
                if "size mismatch" in str(e):
                    if try_resize_vocab(target_path):
                        model = PeftModel.from_pretrained(base_model, target_path)
                    else:
                        raise e
                else:
                    raise e
            
        run_inference(user_input, "关闭客厅的灯")
        
    except Exception as e:
        # import traceback
        # traceback.print_exc()
        print(f"加载失败: {e}")
        print("💡 提示：如果反复失败，请尝试重启脚本。")

print("退出程序。")