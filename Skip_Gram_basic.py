# -*- coding:utf-8 -*-
#! usr/bin/env python3
"""
Created on 10/04/2020 上午12:33
@Author: xinzhi yao
"""

import os
import re
import time
import string
import random
import collections
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Step 1: Data preprocessing.
def str_norm(str_list: list, punc2=' ', num2='NBR', space2=' ', lower=True):
    punctuation = string.punctuation.replace('-', '')
    rep_list = str_list.copy()
    for index, row in enumerate(rep_list):
        row = row.strip()
        row = re.sub("\d+.\d+", num2, row)
        row = re.sub('\d+', num2, row)
        for pun in punctuation:
            row = row.replace(pun, punc2)
        if lower:
            row = row.lower()
        rep_list[index] = re.sub(' +', space2, row)
    return rep_list

def Data_Pre(corpus: str, out: str, head = True):
    if os.path.exists((out)):
        return out
    wf = open(out, 'w', encoding='utf-8')
    with open(corpus, encoding='utf-8') as f:
        if head:
            f.readline()
        for line in f:
            l = line.strip()
            sent_list = str_norm([l], punc2=' ', num2='NBR', space2=' ')
            for sent in sent_list:
                wf.write('{0}\n'.format(sent))
    wf.close()
    return out


raw_file = './data/reference.table.txt'
corpus = Data_Pre(raw_file, './data/corpus.txt')


def read_data(filename: str):
    words = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            l = line.strip().split()
            for word in l:
                words.append(word)
    return words

words = read_data((corpus))
print('Data size: {0} words.'.format(format(len(words), ',')))
"""
Data size: 3,312 words.
"""

# Step 2: Build the dictionary and replace rare words with UNK token
def build_dataset(words, vocabulary_size=40000):
    token_count = [['UNK', -1]]
    token_count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    word2idx = dict()
    data = []
    unk_count = 0

    for word, _ in token_count:
        word2idx[word] = len(word2idx)

    word_set = set(word2idx.keys())

    for word in words:
        if word in word_set:
            index = word2idx[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)

    token_count[0][1] = unk_count

    idx2word = {idx: word for word, idx in word2idx.items()}
    return data, token_count, word2idx, idx2word

vocabulary_size = 40000
data, count, word2idx, idx2word = build_dataset(words, vocabulary_size)
words = list(word2idx.keys())
print('Most common words (+UNK)', count[:6])
print('Sample data: index: {0}, token: {1}'.format(data[:10], [idx2word[i] for i in data[:10] ]))
"""
Most common words
 (+UNK) [['UNK', 0], 
 ('of', 554), ('the', 495), ('and', 398), ('in', 392), ('a', 207)]

Sample data: 
index: [792, 1, 5, 128, 129, 17, 556, 3, 793, 1], 
token: ['Cloning', 'of', 'a', 'cDNA', 'encoding', 'an', 'importin-alpha', 'and', 'down-regulation', 'of']
"""


# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(data, batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  # total window length
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  # 从data开头添加整个窗口长度的idx
  for _ in range(span):
    buffer.append(data[data_index])
    # 防止index 溢出
    data_index = (data_index + 1) % len(data)
    # print(buffer, '\n')
    # print(data[data_index], idx2word[data[data_index]], '\n')
    """
    deque([852], maxlen=9) 
    1 of 
    deque([852, 1], maxlen=9) 
    5 a 
    deque([852, 1, 5], maxlen=9) 
    144 cDNA 
    deque([852, 1, 5, 144], maxlen=9)
    """
    # 0, 1
  for i in range(batch_size // num_skips):
    # center 在这个窗口中的位置
    target = skip_window
    targets_to_avoid = [skip_window]
    # print('i=',i,'target=',target,'targets_to_avoid=',targets_to_avoid,'\n')
    for j in range(num_skips):
        # 窗口中采样非 center的单词
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
        # print(target)
      targets_to_avoid.append(target)
      # print(target,'\t',targets_to_avoid,'\n')
        # index 为 batch 中第几个数据
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]

    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  return batch, labels

data_index = 0
batch_size = 16
# left and right target number.
skip_window = 4
# how many target in window.
num_skips = 8

batch, labels = generate_batch(data=data, batch_size=batch_size,
                               num_skips=num_skips, skip_window=skip_window)

for i in range(16):
  print(batch[i], idx2word[batch[i]],
      '->', labels[i, 0], idx2word[labels[i, 0]])

"""
Cloning of a cDNA 
encoding
an importin-alpha and down-regulation

145 encoding -> 852 Cloning
145 encoding -> 1 of
145 encoding -> 5 a
145 encoding -> 144 cDNA

145 encoding -> 17 an
145 encoding -> 599 importin-alpha
145 encoding -> 3 and
145 encoding -> 853 down-regulation

17 an -> 1 of
17 an -> 5 a
17 an -> 144 cDNA

17 an -> 599 importin-alpha
17 an -> 3 and
17 an -> 853 down-regulation
17 an -> 145 encoding
17 an -> 1 of
"""

# Step 4: Build a skip-gram model.
class SkipGram(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.vocabulary_size = args.vocabulary_size
        self.embedding_size = args.embedding_size

        self.embedding = nn.Embedding(self.vocabulary_size, self.embedding_size) # W = vd lookup  [1*v']*[V*embedding_size]  -> [v* embedding_size]
        self.output = nn.Linear(self.embedding_size, self.vocabulary_size) # 输出层
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):    #x 16
        x = self.embedding(x)  #x  16*128
        x = self.output(x)     #x  16*40000
        log_ps = self.log_softmax(x)  #x 16*40000
        return log_ps

# Step 5: Begin training.
class config():
    def __init__(self):
        self.num_steps = 1000
        self.batch_size = 128
        self.check_step = 20

        self.vocabulary_size = 40000
        self.embedding_size = 200  # Dimension of the embedding vector.
        self.skip_window = 4  # How many words to consider left and right.
        self.num_skips = 8  # How many times to reuse an input to generate a label.

        self.use_cuda = torch.cuda.is_available()

        self.lr = 0.03

args = config()

model = SkipGram(args)
print(model)
"""
SkipGram(
  (embedding): Embedding(40000, 128)
  (output): Linear(in_features=128, out_features=40000, bias=True)
  (log_softmax): LogSoftmax(dim=1)
)
"""

if args.use_cuda:
    model = model.to('cuda')

nll_loss = nn.NLLLoss()
adam_optimizer = optim.Adam(model.parameters(), lr=args.lr)

print('-'*50)
print('Start training.')
average_loss = 0
start_time = time.time()
for step in range(1, args.num_steps):
    batch_inputs, batch_labels = generate_batch(
        data, args.batch_size, args.num_skips, args.skip_window)
    batch_labels = batch_labels.squeeze()
    batch_inputs, batch_labels = torch.LongTensor(batch_inputs), torch.LongTensor(batch_labels)
    if args.use_cuda:
        batch_inputs, batch_labels = batch_inputs.to('cuda'), batch_labels.to('cuda')
    log_ps = model(batch_inputs)
    loss = nll_loss(log_ps, batch_labels)
    average_loss += loss
    adam_optimizer.zero_grad()
    loss.backward()
    adam_optimizer.step()
    if step % args.check_step == 0:
        end_time = time.time()
        average_loss /= args.check_step
        print('Average loss as step {0}: {1:.2f}, cost: {2:.2f}s.'.format(step, average_loss, end_time-start_time))
        start_time = time.time()
        average_loss = 0
"""
Average loss as step 20: 10.33, cost: 30.53s.
Average loss as step 40: 10.11, cost: 4.31s.
Average loss as step 60: 10.26, cost: 4.27s.
Average loss as step 80: 9.96, cost: 4.28s.
Average loss as step 100: 9.99, cost: 4.23s.
Average loss as step 120: 10.34, cost: 4.40s.
Average loss as step 140: 10.47, cost: 4.26s.
Average loss as step 160: 10.53, cost: 4.29s.
Average loss as step 180: 10.88, cost: 4.33s.
"""
print('Training Done.')
print('-'*50)

final_embedding = model.embedding.weight.data
print(final_embedding.shape)
"""
torch.Size([40000, 200])
"""

# Step 6: Visualize the embeddings.
def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
  print('Visualizing.')
  plt.figure(figsize=(18, 18))  #in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i,:]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)
  print('TSNE visualization is completed, saved in {0}.'.format(filename))

matplotlib.use("Agg")
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
plot_only = 500
low_dim_embs = tsne.fit_transform(final_embedding[:plot_only, :])
labels = [ idx2word[i ] for i in range(plot_only) ]
plot_with_labels(low_dim_embs, labels, filename='./data/tsne.png')
