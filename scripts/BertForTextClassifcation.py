import pandas as pd
import numpy as np

from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report

from transformers import BertTokenizerFast, TFBertForSequenceClassification

MAX_TWEET_LENGTH = 20

train = pd.read_csv('./data/full_train_df.csv', index_col=0)
test = pd.read_csv('./data/full_test_df.csv', index_col=0)

x_train = train['tweet'].copy()
y_train = train['label'].copy()

x_test = test['tweet'].copy()
y_test = test['label'].copy()

bt = BertTokenizerFast.from_pretrained('bert-base-uncased')


train_input_ids, train_attention_masks, train_outputs = [], [], []

for sentence, label in zip(x_train, y_train):
    
    sentence_tokens = bt.encode_plus(sentence, max_length=MAX_TWEET_LENGTH, padding='max_length', truncation=True)
    
    train_input_ids.append(np.array(sentence_tokens['input_ids']))
    train_attention_masks.append(np.array(sentence_tokens['attention_mask']))
    train_outputs.append(label)

model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels = 2)
model.compile(optimizer=Adam(learning_rate=.01), loss='binary_crossentropy', metrics=['accuracy'])
model.fit([np.array(train_input_ids), np.array(train_attention_masks)], np.array(train_outputs).reshape(-1, 1), epochs=10, batch_size=32, verbose=2)


test_input_ids, test_attention_masks, test_outputs = [], [], []

bt = BertTokenizerFast.from_pretrained('bert-base-uncased')

for sentence, label in zip(x_test, y_test):
    
    sentence_tokens = bt.encode_plus(sentence, max_length=MAX_TWEET_LENGTH, padding='max_length', truncation=True)
    
    test_input_ids.append(sentence_tokens['input_ids'])
    test_attention_masks.append(sentence_tokens['attention_mask'])
    test_outputs.append(label)

pred = model.predict(x=[np.array(test_input_ids), np.array(test_attention_masks)])

pred_round = pred.round()
print(classification_report(pred_round, test_outputs))

model.save('models/BertForTextClassification')