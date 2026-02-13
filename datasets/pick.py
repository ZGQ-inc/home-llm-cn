import json
import re
import os

INPUT_FILE = 'home_assistant_train_english.jsonl'
OUTPUT_TXT = 'to_translate.txt'

ENTITY_ID_PATTERN = re.compile(r'\b([a-z]+[a-z0-9_]*\.[a-z0-9_]+)\b')

def protect_ids(text):
    return ENTITY_ID_PATTERN.sub(r'`\1`', text)

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"错误: 找不到文件 {INPUT_FILE}，请下载：https://huggingface.co/datasets/acon96/Home-Assistant-Requests-V2")
        return

    print(f"正在提取文本到 {OUTPUT_TXT} ...")
    
    count = 0
    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_TXT, 'w', encoding='utf-8') as f_out:
        
        for line_idx, line in enumerate(f_in):
            if not line.strip(): continue
            try:
                data = json.loads(line)
                if 'messages' not in data: continue

                for msg_idx, msg in enumerate(data['messages']):
                    role = msg.get('role')
                    content = msg.get('content')

                    if role in ['user', 'assistant']:
                        text_content = ""
                        
                        if isinstance(content, list):
                            for item in content:
                                if item.get('type') == 'text':
                                    text_content = item['text']
                                    break
                        elif isinstance(content, str):
                            text_content = content

                        if text_content and not text_content.strip().startswith('{'):
                            f_out.write(f"[[ID:{line_idx}|{msg_idx}]]\n")
                            
                            protected_text = protect_ids(text_content)
                            f_out.write(f"{protected_text}\n")
                            
                            f_out.write("==========\n")
                            count += 1
                
            except json.JSONDecodeError:
                pass

    print(f"提取完成。共提取了 {count} 个文本块。")
    print(f"请使用 translate.py 翻译 '{OUTPUT_TXT}'。")

if __name__ == "__main__":
    main()
