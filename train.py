# train.py

import os
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Dropout, Reshape, add, concatenate
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import warnings

warnings.filterwarnings('ignore')
mlflow.tensorflow.autolog(disable=True)

mlflow.set_experiment("Image Captioning")

# Paths
image_path = '/home/soumik/SPE_Final_Project/data/flickr/Images'
captions_file = '/home/soumik/SPE_Final_Project/data/flickr/captions.txt'
model_dir = 'models/'
os.makedirs(model_dir, exist_ok=True)
feature_file = os.path.join(model_dir, 'features.pkl')

# Read and preprocess captions
data = pd.read_csv(captions_file)

def text_preprocessing(data):
    data['caption'] = data['caption'].apply(lambda x: x.lower())
    data['caption'] = data['caption'].str.replace("[^A-Za-z]", " ", regex=True)
    data['caption'] = data['caption'].str.replace("\s+", " ", regex=True)
    data['caption'] = data['caption'].apply(lambda x: " ".join([word for word in x.split() if len(word) > 1]))
    data['caption'] = "startseq " + data['caption'] + " endseq"
    return data

data = text_preprocessing(data)
captions = data['caption'].tolist()

# Tokenizer and max length
tokenizer = Tokenizer()
tokenizer.fit_on_texts(captions)
vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(caption.split()) for caption in captions)

# Save tokenizer early
with open(os.path.join(model_dir, 'tokenizer.pkl'), 'wb') as f:
    pickle.dump(tokenizer, f)
with open(os.path.join(model_dir, 'max_length.txt'), 'w') as f:
    f.write(str(max_length))

# 80% Train, 10% Validation, 10% Test
images = data['image'].unique().tolist()
nimages = len(images)
split_1 = int(0.8 * nimages)
split_2 = int(0.9 * nimages)

train_images = images[:split_1]
val_images = images[split_1:split_2]
test_images = images[split_2:]

train_df = data[data['image'].isin(train_images)].reset_index(drop=True)
val_df = data[data['image'].isin(val_images)].reset_index(drop=True)
test_df = data[data['image'].isin(test_images)].reset_index(drop=True)

# Load precomputed features
with open(feature_file, 'rb') as f:
    features = pickle.load(f)

# Data generator
class CustomDataGenerator(Sequence):
    def __init__(self, df, X_col, y_col, batch_size, tokenizer, vocab_size, max_length, features, shuffle=True):
        self.df = df.copy()
        self.X_col = X_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.features = features
        self.shuffle = shuffle
        self.n = len(self.df)

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __len__(self):
        return self.n // self.batch_size

    def __getitem__(self, index):
        batch = self.df.iloc[index * self.batch_size:(index + 1) * self.batch_size]
        return self.__get_data(batch)

    def __get_data(self, batch):
        X1, X2, y = [], [], []
        for _, row in batch.iterrows():
            image = row[self.X_col]
            caption = row[self.y_col]
            feature = self.features[image][0]
            seq = self.tokenizer.texts_to_sequences([caption])[0]
            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]
                in_seq = pad_sequences([in_seq], maxlen=self.max_length)[0]
                out_seq = to_categorical([out_seq], num_classes=self.vocab_size)[0]
                X1.append(feature)
                X2.append(in_seq)
                y.append(out_seq)
        return (np.array(X1), np.array(X2)), np.array(y)

# Generators
train_gen = CustomDataGenerator(train_df, 'image', 'caption', 64, tokenizer, vocab_size, max_length, features)
val_gen = CustomDataGenerator(val_df, 'image', 'caption', 64, tokenizer, vocab_size, max_length, features)

# Model
input1 = Input(shape=(1920,))
input2 = Input(shape=(max_length,))
img_features = Dense(256, activation='relu')(input1)
img_features_reshaped = Reshape((1, 256))(img_features)

sentence_features = Embedding(vocab_size, 256, mask_zero=False)(input2)
merged = concatenate([img_features_reshaped, sentence_features], axis=1)
sentence_features = LSTM(256)(merged)

x = Dropout(0.5)(sentence_features)
x = add([x, img_features])
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(vocab_size, activation='softmax')(x)

caption_model = Model(inputs=[input1, input2], outputs=output)
caption_model.compile(loss='categorical_crossentropy', optimizer='adam')

# Callbacks
checkpoint = ModelCheckpoint(os.path.join(model_dir, "model.h5"), monitor="val_loss", save_best_only=True, mode="min", verbose=1)
earlystop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.2, min_lr=1e-8, verbose=1)

# MLflow logging
with mlflow.start_run():
    # Log hyperparameters
    mlflow.log_param("batch_size", 64)
    mlflow.log_param("optimizer", "adam")
    mlflow.log_param("embedding_dim", 256)
    mlflow.log_param("lstm_units", 256)
    mlflow.log_param("dropout", 0.5)
    mlflow.log_param("epochs", 7)

    # Train and capture history
    history = caption_model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=7,
        callbacks=[checkpoint, earlystop, reduce_lr],
        verbose=1
    )

    # Log metrics per epoch
    for epoch in range(len(history.history['loss'])):
        mlflow.log_metric("loss", history.history['loss'][epoch], step=epoch)
        mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)

    # Save training curve
    plt.figure()
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Curve')
    plot_path = os.path.join(model_dir, 'training_curve.png')
    plt.savefig(plot_path)
    mlflow.log_artifact(plot_path)

