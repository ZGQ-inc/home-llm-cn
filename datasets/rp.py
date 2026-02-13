import json
import re
import os

ORIGINAL_JSONL = 'home_assistant_train_english.jsonl'
TRANSLATED_TXT = 'translated.txt'
FINAL_OUTPUT = 'home_assistant_train_chinese.jsonl'

def parse_txt_file(txt_file):
    mapping = {}
    print("正在解析...")
    
    with open(txt_file, 'r', encoding='utf-8') as f:
        content = f.read()

    blocks = content.split('==========')
    
    id_pattern = re.compile(r'\[\[\s*ID\s*:\s*(\d+)\s*\|\s*(\d+)\s*\]\]', re.IGNORECASE)

    for block in blocks:
        block = block.strip()
        if not block: continue
        
        match = id_pattern.search(block)
        if match:
            line_idx = int(match.group(1))
            msg_idx = int(match.group(2))
            lines = block.split('\n')
            text_lines = []
            start_collecting = False
            for line in lines:
                if start_collecting:
                    text_lines.append(line)
                elif id_pattern.search(line):
                    start_collecting = True
            
            clean_text = '\n'.join(text_lines).strip()
            
            clean_text = clean_text.replace('`', '')
            
            mapping[(line_idx, msg_idx)] = clean_text
            
    print(f"解析成功：获取了 {len(mapping)} 条翻译记录。")
    return mapping

def main():
    if not os.path.exists(TRANSLATED_TXT):
        print(f"错误: 找不到翻译文件 {TRANSLATED_TXT}")
        return

    translation_map = parse_txt_file(TRANSLATED_TXT)
    
    print("正在生成数据集...")
    success_count = 0
    
    with open(ORIGINAL_JSONL, 'r', encoding='utf-8') as f_in, \
         open(FINAL_OUTPUT, 'w', encoding='utf-8') as f_out:
        
        for line_idx, line in enumerate(f_in):
            if not line.strip(): continue
            
            try:
                data = json.loads(line)
                
                if 'messages' in data:
                    for msg_idx, msg in enumerate(data['messages']):
                        key = (line_idx, msg_idx)
                        
                        if key in translation_map:
                            translated_text = translation_map[key]
                            
                            if isinstance(msg['content'], str):
                                msg['content'] = translated_text
                            elif isinstance(msg['content'], list):
                                for item in msg['content']:
                                    if item.get('type') == 'text':
                                        item['text'] = translated_text
                            
                            success_count += 1
                
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError:
                pass

    print(f"完成。成功替换了 {success_count} 处文本。")
    print(f"最终文件: {FINAL_OUTPUT}")

if __name__ == "__main__":
    main()
