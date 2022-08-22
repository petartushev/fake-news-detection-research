import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, Flatten

from transformers import BertTokenizerFast, TFBertModel

MAX_TWEET_LENGTH = 240

train = pd.read_csv('./data/full_train_df.csv', index_col=0)

x_train = train['tweet'].copy()
y_train = train['label'].copy()

bt = BertTokenizerFast.from_pretrained('bert-base-uncased')

train_input_ids, train_attention_masks, train_outputs = [], [], []

for sentence, label in zip(x_train, y_train):
    
    sentence_tokens = bt.encode_plus(sentence, max_length=MAX_TWEET_LENGTH, padding='max_length', truncation=True)
    
    train_input_ids.append(np.array(sentence_tokens['input_ids']))
    train_attention_masks.append(np.array(sentence_tokens['attention_mask']))
    train_outputs.append(label)



bert_base_uncased = TFBertModel.from_pretrained('bert-base-uncased')

input_ids = Input(shape=(MAX_TWEET_LENGTH,), name='input_ids', dtype='int32')
att_masks = Input(shape=(MAX_TWEET_LENGTH,), name='masked_tokens', dtype='int32')

bert_in = bert_base_uncased(input_ids, attention_mask=att_masks)[1]

dense_1 = Dense(64, name='dense_1', activation=tf.keras.layers.LeakyReLU(alpha=.1))(bert_in)

dense_2 = Dense(32, name='dense_2', activation=tf.keras.layers.LeakyReLU(alpha=.1))(dense_1)

flatten = Flatten()(dense_2)

out = Dense(1, name='output', activation='sigmoid')(flatten)

bert_base_model = Model(inputs=[input_ids, att_masks], outputs=[out])

bert_base_model.compile(optimizer=Adam(learning_rate=.01), loss='binary_crossentropy', metrics=['accuracy'])

bert_base_model.fit([np.array(train_input_ids), np.array(train_attention_masks)], np.array(train_outputs).reshape(-1, 1), epochs=10, batch_size=16, verbose=2, use_multiprocessing=True)

bert_base_model.save('models/BERT_base_integrated')