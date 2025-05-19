import os
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications import DenseNet201
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import warnings

warnings.filterwarnings('ignore')
nltk.download('punkt')

# Paths
image_path = 'data/flickr8k/Images'
captions_file = 'data/flickr8k/captions.txt'
model_path = 'models/model.h5'
features_file = "models/features.pkl"

# Read captions
data = pd.read_csv(captions_file)

# Preprocess captions
def text_preprocessing(data):
    data['caption'] = data['caption'].apply(lambda x: x.lower())
    data['caption'] = data['caption'].str.replace("[^A-Za-z]", " ", regex=True)
    data['caption'] = data['caption'].str.replace("\s+", " ", regex=True)
    data['caption'] = data['caption'].apply(lambda x: " ".join([word for word in x.split() if len(word) > 1]))
    data['caption'] = "startseq " + data['caption'] + " endseq"
    return data

data = text_preprocessing(data)

# Tokenizer
captions = data['caption'].tolist()
tokenizer = Tokenizer()
tokenizer.fit_on_texts(captions)
vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(caption.split()) for caption in captions)

# Train/Val/Test Split
images = data['image'].unique().tolist()
nimages = len(images)
split_1 = int(0.8 * nimages)
split_2 = int(0.9 * nimages)

test_images = images[split_2:]
test = data[data['image'].isin(test_images)].reset_index(drop=True)

# Load precomputed features
with open(features_file, 'rb') as f:
    features = pickle.load(f)


# Load trained caption model
caption_model = load_model(model_path)

# Reverse word index
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Predict caption for one image
def predict_caption(model, image, tokenizer, max_length, features):
    feature = features[image]
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
    return in_text

# Evaluate using BLEU score
smoothie = SmoothingFunction().method4
bleu_scores = []
predictions = []

for index, row in tqdm(test.iterrows(), total=len(test), desc="Evaluating"):
    ground_truth = row['caption'].split()
    predicted_caption = predict_caption(caption_model, row['image'], tokenizer, max_length, features)
    predicted_caption_tokens = predicted_caption.split()[1:-1]  # remove startseq, endseq
    bleu = sentence_bleu([ground_truth[1:-1]], predicted_caption_tokens, smoothing_function=smoothie)
    bleu_scores.append(bleu)
    predictions.append(predicted_caption)

test['predicted_caption'] = predictions
print("\nAverage BLEU score on test set:", np.mean(bleu_scores))


