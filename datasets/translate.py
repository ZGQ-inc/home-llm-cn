import requests
import json
import os
import sys
from tqdm import tqdm

INPUT_FILE = "to_translate.txt"
OUTPUT_FILE = "translated.txt"
MODEL_NAME = "translategemma:12b"
API_URL = "http://localhost:11434/api/generate"

DEBUG_PRINT = True 

def translate_text(text):
    if not text.strip():
        return text

    prompt = (
        f"Translate the following text into Simplified Chinese. "
        f"Keep the original meaning strictly. Do not output any explanation or notes.\n\n"
        f"Text: {text}\nTranslation:"
    )
    
    data = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_ctx": 4096
        }
    }

    try:
        response = requests.post(API_URL, json=data)
        if response.status_code == 200:
            result = response.json()
            translated = result.get("response", "").strip()
            return translated
        else:
            return f"[Error: API {response.status_code}] {text}"
    except Exception as e:
        return f"[Error: Connection] {text}"

def is_metadata(line):
    stripped = line.strip()
    if stripped.startswith("[[ID:") and stripped.endswith("]]"):
        return True
    if stripped == "==========":
        return True
    return False

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"错误: 找不到文件 {INPUT_FILE}")
        return

    print(f"正在扫描文件行数...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    print(f"开始任务: {MODEL_NAME} | 总行数: {total_lines}")
    print(f"输出文件: {OUTPUT_FILE}")

    buffer_text = []
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        
        progress_bar = tqdm(f_in, total=total_lines, unit="line", desc="Translation Progress")
        
        for line in progress_bar:
            stripped_line = line.strip()

            if is_metadata(stripped_line):
                if buffer_text:
                    source_text = "\n".join(buffer_text)
                    translated_text = translate_text(source_text)
                    
                    f_out.write(translated_text + "\n")
                    
                    if DEBUG_PRINT:
                        tqdm.write(f"\n[原文]: {source_text}")
                        tqdm.write(f"[译文]: {translated_text}")
                        tqdm.write(f"-----------------------------")
                    
                    buffer_text = []

                f_out.write(line)
            
            else:
                if stripped_line:
                    buffer_text.append(stripped_line)
                else:
                    if buffer_text:
                        source_text = "\n".join(buffer_text)
                        translated_text = translate_text(source_text)
                        f_out.write(translated_text + "\n")
                        if DEBUG_PRINT:
                            tqdm.write(f"\n[原文]: {source_text}")
                            tqdm.write(f"[译文]: {translated_text}")
                            tqdm.write(f"-----------------------------")
                        buffer_text = []
                    f_out.write(line)

        if buffer_text:
            source_text = "\n".join(buffer_text)
            translated_text = translate_text(source_text)
            f_out.write(translated_text + "\n")
            if DEBUG_PRINT:
                tqdm.write(f"\n[原文]: {source_text}")
                tqdm.write(f"[译文]: {translated_text}")

    print(f"\n任务完成。")

if __name__ == "__main__":
    main()