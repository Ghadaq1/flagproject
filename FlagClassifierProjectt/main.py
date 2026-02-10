from data_loader import load_data
from cnn_feature_extractor import build_cnn
from svm_trainer import train_svm
import numpy as np
import tensorflow as tf


# Step 1: Load data
train_gen, val_gen = load_data('path_to_flag_images', img_size=(128, 128))

# Step 2: Build CNN model
cnn_model = build_cnn()
train_features = cnn_model.predict(train_gen)
val_features = cnn_model.predict(val_gen)

# Step 3: Get labels
y_train = train_gen.classes
y_val = val_gen.classes

# Step 4: Train SVM
svm, label_encoder = train_svm(train_features, y_train, val_features, y_val)
