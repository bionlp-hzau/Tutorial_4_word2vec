# -*- coding:utf-8 -*-
#! usr/bin/env python3
"""
Created on 10/04/2020 上午12:33
@Author: xinzhi yao
"""

import collections
import string
import os
import re
import time
import random

from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
from collections import Counter

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
    wf = open(out, 'w')
    with open(corpus) as f:
        if head:
            l = f.readline()
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
print('Data size: {0}.'.format(format(len(words), ',')))


# Step 2: Build the dictionary and replace rare words with UNK token
def build_dataset(words, vocabulary_size=40000):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    # word2index
    for word, _ in count:
        dictionary[word] = len(dictionary)
    # data转为index表示
    data = []
    unk_count = 0
    word_set = set(dictionary.keys())
    for word in words:
        if word in word_set:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    # index2word
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

vocabulary_size = 40000
data, count, dictionary, reverse_dictionary = build_dataset(words, vocabulary_size)
words = list(dictionary.keys())
print('Most common words (+UNK)', count[:6])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(data, batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1 # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
    #print(buffer,'\n')
    #print(data_index, reverse_dictionary[data_index], '\n')
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [ skip_window ]
    #print('i=',i,'target=',target,'targets_to_avoid=',targets_to_avoid,'\n')
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      #print(target,'\t',targets_to_avoid,'\n')
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  return batch, labels

data_index = 0

batch, labels = generate_batch(data=data, batch_size=16, num_skips=8, skip_window=4)
for i in range(16):
  print(batch[i], reverse_dictionary[batch[i]],
      '->', labels[i, 0], reverse_dictionary[labels[i, 0]])
# 6 rice -> 4 in
# 6 rice -> 13 by
# 6 rice -> 108 leaves
# 6 rice -> 5 nbr
# 6 rice -> 368 light
# 6 rice -> 16 gene
# 6 rice -> 877 central
# 6 rice -> 16 gene
# 108 leaves -> 16 gene
# 108 leaves -> 13 by
# 108 leaves -> 5 nbr
# 108 leaves -> 77 research
# 108 leaves -> 368 light
# 108 leaves -> 6 rice
# 108 leaves -> 4 in
# 108 leaves -> 877 central



# Step 4: Build a skip-gram model.
class SkipGram(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.vocabulary_size = args.vocabulary_size
        self.embedding_size = args.embedding_size

        self.embedding = nn.Embedding(self.vocabulary_size, self.embedding_size) # W = vd lookup  [1*v']*[V*embedding_size]  -> [v* embedding_size]
        self.output = nn.Linear(self.embedding_size, self.vocabulary_size) # 输出层
        self.log_softmax = nn.LogSoftmax(dim=1)

    def similarity_eval(self, embedding, valid_data):
        embedding = F.normalize(embedding)
        valid_embeddings = self.embedding(valid_data)
        valid_embeddings = F.normalize(valid_embeddings)
        similarity = valid_embeddings.mm(embedding.t())
        return similarity

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
        self.eval_step = 300


        self.vocabulary_size = 40000
        self.embedding_size = 128  # Dimension of the embedding vector.
        self.skip_window = 1  # How many words to consider left and right.
        self.num_skips = 2  # How many times to reuse an input to generate a label.

        # We pick a random validation set to sample nearest neighbors. Here we limit the
        # validation samples to the words that have a low numeric ID, which by
        # construction are also the most frequent.
        self.valid_size = 16  # Random set of words to evaluate similarity on.
        self.valid_window = 100  # Only pick dev samples in the head of the distribution.
        self.num_sampled = 64  # Number of negative examples to sample.
        self.use_cuda = torch.cuda.is_available()

        self.lr = 0.03
        self.momentum = 0

args = config()


model = SkipGram(args)
print(model)

if args.use_cuda:
    model = model.to('cuda')

criterion = nn.NLLLoss() #-(log softmax(x)) * [class]
optimizer = optim.Adam(model.parameters(), lr=args.lr)

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

    # # fixme:
    # break
    # input, output, and noise vectors
    log_ps = model(batch_inputs)
    loss = criterion(log_ps, batch_labels)

    average_loss += loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % args.check_step == 0:
        end_time = time.time()
        average_loss /= args.check_step
        # The average loss is an estimate of the loss over the last 2000 batches.
        print('Average loss as step {0}: {1:.2f}, cost: {2:.2f}s.'.format(step, average_loss, end_time-start_time))
        start_time = time.time()
        average_loss = 0

    if step % args.eval_step == 0:
        print('-'*50)
        print('Evaluation.')
        model.eval()

        valid_examples = np.random.choice(args.valid_window, args.valid_size, replace=False)
        valid_examples = torch.LongTensor(valid_examples)
        if args.use_cuda:
            valid_examples = valid_examples.to('cuda')

        # valid_size, embedding_size
        Sim = model.similarity_eval(model.embedding.weight.data, valid_examples)
        for i in range(args.valid_size):
            # 换成单词
            valid_word = reverse_dictionary[int(valid_examples[i].data)]
            # 拿出最接近的８个次
            top_k = 8
            nearest = (-Sim[i, :]).argsort()[1: top_k+1]
            log_str = "Nearest to: '{0}' is ".format(valid_word)
            for k in range(top_k):
                close_word = reverse_dictionary[int(nearest[k])]
                log_str = "{0} {1},".format(log_str, close_word)
            print(log_str)
        model.train()
        print('-'*50)
print('Training Done.')
print('-'*50)
final_embedding = model.embedding.weight.data #40000 *128

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
labels = [reverse_dictionary[i] for i in range(plot_only)]
plot_with_labels(low_dim_embs, labels, filename='./datatsne.png')
