import requests

# 莎士比亚数据集地址
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
response = requests.get(url)

# 保存到本地
with open('data/Shakespear.txt', 'w', encoding='utf-8') as f:
    f.write(response.text)

print("下载完成！文件名: input.txt")