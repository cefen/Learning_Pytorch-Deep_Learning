import os 
import pickle

meta_path = 'meta.pkl'
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    vocab_size = meta['vocab_size']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    print(f"成功加载词表，大小为: {vocab_size}")
# 在你的主程序中加入这几行测试
test_str = "郭靖"
encoded = encode(test_str)
decoded = decode(encoded)
print(f"原始: {test_str} -> 编码: {encoded} -> 解码: {decoded}")