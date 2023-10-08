# -*- coding: utf-8 -*-
"""Chest Cancer Detection.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1cs7OvCh9vbmBNefma70HjVJ9ANJGymxi
"""

import zipfile
import os

# Define the path to the uploaded zip file and the extraction directory
zip_file_path = '/content/Chest Xray.zip'
extraction_dir = '/mnt/data/chest_cancer_data/'

# Create the extraction directory if it doesn't exist
if not os.path.exists(extraction_dir):
    os.makedirs(extraction_dir)

# Unzip the file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extraction_dir)

# List the contents of the extraction directory
extracted_files = os.listdir(extraction_dir)
extracted_files

# Explore the contents of the "Data" directory
data_dir = os.path.join(extraction_dir, 'Data')
data_contents = os.listdir(data_dir)
data_contents

import os

data_dir = os.path.join(extraction_dir, 'Data')

# Get the paths of subdirectories in the "Data" directory
subdirectories = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

# Now, 'subdirectories' contains the paths of subdirectories within the "Data" directory
print(subdirectories)

# Explore the contents of the "train", "test", and "valid" directories
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
valid_dir = os.path.join(data_dir, 'valid')

# List the categories present in each directory
train_categories = os.listdir(train_dir)
test_categories = os.listdir(test_dir)
valid_categories = os.listdir(valid_dir)

train_categories, test_categories, valid_categories

"""The "Data" directory contains three subdirectories: "train", "test", and "valid". These likely correspond to the training, testing, and validation datasets. Let's explore each one to get a better understanding of how the data is organized"""

# Function to count the number of files in each category for a given directory
def count_files_in_categories(base_dir, categories):
    count_dict = {}
    for category in categories:
        category_path = os.path.join(base_dir, category)
        num_files = len([f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))])
        count_dict[category] = num_files
    return count_dict

# Count the number of files in each category for train, test, and valid directories
train_counts = count_files_in_categories(train_dir, train_categories)
test_counts = count_files_in_categories(test_dir, test_categories)
valid_counts = count_files_in_categories(valid_dir, valid_categories)

train_counts, test_counts, valid_counts

"""Here is the distribution of the number of images in each category for the train, test, and validation datasets:

1. Training set (train)

Adenocarcinoma: 195 images

Large Cell Carcinoma: 115 images

Normal: 148 images

Squamous Cell Carcinoma: 155 images

2. Testing set (test)

Adenocarcinoma: 120 images

Large Cell Carcinoma: 51 images

Normal: 54 images

Squamous Cell Carcinoma: 90 images

3. Validation set (valid)

Adenocarcinoma: 23 images

Large Cell Carcinoma: 21 images

Normal: 13 images

Squamous Cell Carcinoma: 15 images

# Preprocessing
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# Image dimensions
img_width, img_height = 150, 150

# Initialize data generators for training, validation, and test sets
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

valid_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Create data generators
train_generator = train_datagen.flow_from_directory(
    '/mnt/data/chest_cancer_data/Data/train' ,  # Replace with your training directory path
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical'
)

valid_generator = valid_datagen.flow_from_directory(
    '/mnt/data/chest_cancer_data/Data/valid',  # Replace with your validation directory path
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    '/mnt/data/chest_cancer_data/Data/test',  # Replace with your test directory path
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Build the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(4))  # Number of classes
model.add(Activation('softmax'))

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.0001),
    metrics=['accuracy']
)

# Train the model
model.fit(
    train_generator,
    epochs=30,
    validation_data=valid_generator
)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")