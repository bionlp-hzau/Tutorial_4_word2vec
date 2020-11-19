#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Created on 19/11/2020 14:21
@Author: XinZhi Yao
"""

from transformers import BertTokenizer, BertModel

# 1. Load model.
model_name = 'bert-base-uncased'
model = BertModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# 2. Data preprocessing.
token = 'Harry'
token_input = tokenizer(token, return_tensors='pt')
token_decode = tokenizer.decode(token_input['input_ids'][0])
"""
token_input:
    {'input_ids': [101, 4302, 102],
    'token_type_ids': [0, 0, 0],
    'attention_mask': [1, 1, 1]}
token_decode:
    '[CLS] harry [SEP]'
"""

# 3. Calculate word embedding
model.eval()
token_embedding, _ = model(**token_input)
print(token_embedding.shape)
"""
torch.Size([1, 3, 768])
"""
token_embedding = token_embedding.squeeze(0)
print(token_embedding.shape)
"""
torch.Size([3, 768])
"""
# without_speical_token
token_embedding = token_embedding[1, :]
print(token_embedding.shape)
"""
torch.Size([768])
"""

# 4. Save embedding.
token_embedding = ' '.join([str(i) for i in token_embedding.detach().numpy().tolist()])
embedding_file = 'data/Bert_embedding.txt'
with open(embedding_file, 'w') as wf:
    wf.write('{0}\t{1}\n'
             .format(token, token_embedding))
print(token_embedding)
'''
'0.0815548524260521 -0.4707965552806854  ...   0.396944522857666 -0.16670702397823334'
'''
