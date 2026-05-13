import csv
import os
import random

# --- 配置区 ---
input_dir = "E:/Programs/WeChatData/texts"
output_file = 'E:/projects/LearningWuEnda/data/all_my_talks.txt'
my_sender_value = '1'  # 通常 1 代表你自己发的
trash_words = ["[", "]", "红包", "https", "http"]

all_my_messages = []
file_count = 0

print("开始扫描文件夹...")

# 遍历文件夹下所有文件
for filename in os.listdir(input_dir):
    if filename.endswith('.csv'):
        file_path = os.path.join(input_dir, filename)
        file_count += 1
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # 使用 DictReader 自动处理列名
                reader = csv.DictReader(f)
                
                for row in reader:
                    # 1. 判定发送者 (确保这里的 'is_sender' 和你 CSV 里的表头对得上)
                    if row.get('is_sender') == my_sender_value:
                        content = row.get('msg', '')
                        
                        # 2. 清洗逻辑
                        content = content.lstrip('\n').strip()
                        
                        # 3. 过滤垃圾信息
                        if not content: continue
                        if any(word in content for word in trash_words): continue
                        
                        all_my_messages.append(content)
                        
        except Exception as e:
            print(f"读取文件 {filename} 时出错: {e}")

# --- 汇总写入 ---
with open(output_file, 'w', encoding='utf-8') as out:
    random.shuffle(all_my_messages)
    for msg in all_my_messages:
        out.write(msg + '\n')

print(f"处理完成！")
print(f"扫描文件数: {file_count}")
print(f"提取金句总数: {len(all_my_messages)}")
print(f"最终语料库已保存至: {output_file}")