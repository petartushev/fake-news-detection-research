import pandas as pd
import numpy as np
import os

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, Flatten

from transformers import DistilBertTokenizer, TFDistilBertModel

MAX_TWEET_LENGTH = 240

train = pd.read_csv(os.getcwd() + '/data/full_train_df.csv', index_col=0)

x_train = train['tweet'].copy()
y_train = train['label'].copy()

bt = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


train_input_ids, train_attention_masks, train_outputs = [], [], []

for sentence, label in zip(x_train, y_train):
    
    sentence_tokens = bt.encode_plus(sentence, max_length=MAX_TWEET_LENGTH, padding='max_length', truncation=True)
    
    train_input_ids.append(np.array(sentence_tokens['input_ids']))
    train_attention_masks.append(np.array(sentence_tokens['attention_mask']))
    train_outputs.append(label)


distilbert_base_uncased = TFDistilBertModel.from_pretrained('distilbert-base-uncased')

input_ids = Input(shape=(MAX_TWEET_LENGTH,), name='input_ids', dtype='int32')
att_masks = Input(shape=(MAX_TWEET_LENGTH,), name='masked_tokens', dtype='int32')

bert_in = distilbert_base_uncased(input_ids, attention_mask=att_masks)[0]

dense_1 = Dense(64, name='dense_1', activation=tf.keras.layers.LeakyReLU(alpha=.1))(bert_in)

dense_2 = Dense(32, name='dense_1', activation=tf.keras.layers.LeakyReLU(alpha=.1))(dense_1)

flatten = Flatten()(dense_1)

out = Dense(1, name='output', activation='sigmoid')(flatten)

distilbert_base_model = Model(inputs=[input_ids, att_masks], outputs=[out])

distilbert_base_model.compile(optimizer=Adam(learning_rate=.01), loss='binary_crossentropy', metrics=['accuracy'])

distilbert_base_model.fit([np.array(train_input_ids), np.array(train_attention_masks)], np.array(train_outputs).reshape(-1, 1), epochs=10, batch_size=16, verbose=2, use_multiprocessing=True)

distilbert_base_model.save('models/distilBERT_base_integrated')