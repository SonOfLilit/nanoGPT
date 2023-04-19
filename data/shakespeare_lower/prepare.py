import sys
import os
import requests
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import lowerizer

# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r') as f:
    data = f.read()
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with lowerizer
train_ids, train_flags = lowerizer.encode(train_data)
val_ids, val_flags = lowerizer.encode(val_data)
val_decoded = lowerizer.decode(val_ids, val_flags)
assert val_data == val_decoded, ((val_data[:100], val_decoded[:100]), (val_data[-100:], val_decoded[-100:]))
print(f"train has {len(train_ids):,} tokens, {len(set(train_ids)):,} unique")
print(f"val has {len(val_ids):,} tokens, {len(set(val_ids)):,} unique")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
train_flags = np.array(train_flags, dtype=np.bool8)
val_ids = np.array(val_ids, dtype=np.uint16)
val_flags = np.array(val_flags, dtype=np.bool8)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
train_flags.tofile(os.path.join(os.path.dirname(__file__), 'train_flags.bin'))
val_flags.tofile(os.path.join(os.path.dirname(__file__), 'val_flags.bin'))

# train.bin has 301,966 tokens
# val.bin has 36,059 tokens
