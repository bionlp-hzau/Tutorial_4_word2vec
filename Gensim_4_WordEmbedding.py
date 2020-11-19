#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Created on 19/11/2020 14:45 
@Author: XinZhi Yao
Documentation: https://radimrehurek.com/gensim/models/word2vec.html
"""

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

# 1. data path
corpus_file = 'data/corpus.txt'
model_save_file = 'data/Gensim.model.bin'
embedding_save_file = 'data/Gensim.embedding.txt'

# 2. Define hyper-parameters
# Dimensionality of the word vectors.
size = 100
# Maximum distance between the current and predicted word within a sentence.
window = 5
# Ignores all words with total frequency lower than this.
min_count = 5
# Training algorithm: 1 for skip-gram; otherwise CBOW.
sg = 1
# The initial learning rate.
alpha = 0.02
# Number of iterations (epochs) over the corpus.
epochs = 2
# Target size (in words) for batches of examples passed to worker threads.
batch_words = 10000

# 3. Training model
model = Word2Vec(LineSentence(corpus_file), size=size, window=window,
                 min_count=min_count, sg=sg, alpha=alpha, iter=epochs,
                 batch_words=batch_words)

# 4. Save Model and Embedding.
# full Word2Vec object state
model.save(model_save_file)
# just the KeyedVectors.
model.wv.save_word2vec_format(embedding_save_file, binary=False)

