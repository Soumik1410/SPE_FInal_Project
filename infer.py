# infer_utils.py

import os
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import DenseNet201
import warnings

warnings.filterwarnings('ignore')

# Paths
model_dir = 'models'
model_path = os.path.join(model_dir, 'model.h5')
tokenizer_path = os.path.join(model_dir, 'tokenizer.pkl')
img_size = 224

# Load tokenizer and model once
with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)

vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(caption.split()) for caption in tokenizer.index_word.values())

model = load_model(model_path)
feature_extractor = DenseNet201(include_top=False, pooling='avg')

def extract_feature_from_array(img_array):
    img = img_array / 255.
    img = np.expand_dims(img, axis=0)
    feature = feature_extractor.predict(img, verbose=0)
    return feature

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_caption(img_array):
    feature = extract_feature_from_array(img_array)
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        y_pred = model.predict([feature, sequence], verbose=0)
        y_pred = np.argmax(y_pred)
        word = idx_to_word(y_pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return ' '.join(in_text.split()[1:-1])

