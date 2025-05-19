import os
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.models import Model

image_path = 'data/flickr/Images'
captions_file = 'data/flickr/captions.txt'
model_dir = 'models/'
img_size = 224

data = pd.read_csv(captions_file)

feature_extractor = DenseNet201(include_top=False, pooling='avg')
features = {}
for image in tqdm(data['image'].unique().tolist(), desc="Extracting image features"):
    img = load_img(os.path.join(image_path, image), target_size=(img_size, img_size))
    img = img_to_array(img) / 255.
    img = np.expand_dims(img, axis=0)
    feature = feature_extractor.predict(img, verbose=0)
    features[image] = feature

with open(os.path.join(model_dir, "features.pkl"), "wb") as f:
    pickle.dump(features, f)

print("Feature extraction complete and saved to models/features.pkl")

