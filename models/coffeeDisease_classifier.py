
#Coffee disease detector

import pandas as pd
import os
from sklearn.model_selection import train_test_split
import numpy as np

#Loading the csv file
csv_file = r"E:/A2SV Hackathon/coffee leaf diseases/train_classes.csv"
data = pd.read_csv(csv_file)

image_dir = r"E:/A2SV Hackathon/coffee leaf diseases/images"
image_paths = [os.path.join(image_dir, f"{img_id}.jpg") for img_id in data['id'].values]

# Extracting labels
labels = data[['miner', 'rust', 'phoma']].values

# Split data into training and validation sets
train_image_paths, val_image_paths, train_labels, val_labels = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42
)

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence

class CustomDataGenerator(Sequence):
    def __init__(self, image_paths, labels, batch_size, image_size, augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.image_size = image_size
        self.augment = augment
        self.datagen = ImageDataGenerator(rescale=1.0/255.0)
        if self.augment:
            self.datagen = ImageDataGenerator(
                rescale=1.0/255.0,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.image_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        images = np.array([self.load_image(file_path) for file_path in batch_x])
        return images, batch_y

    def load_image(self, file_path):
        image = tf.keras.preprocessing.image.load_img(file_path, target_size=self.image_size)
        image = tf.keras.preprocessing.image.img_to_array(image)
        return image

    def on_epoch_end(self):
        # Shuffle data at the end of each epoch
        indices = np.arange(len(self.image_paths))
        np.random.shuffle(indices)
        self.image_paths = self.image_paths[indices]
        self.labels = self.labels[indices]

# Define constants
IMAGE_WIDTH, IMAGE_HEIGHT = 150, 150
BATCH_SIZE = 32
EPOCHS = 15

# Create data generators
train_generator = CustomDataGenerator(train_image_paths, train_labels, BATCH_SIZE, (IMAGE_WIDTH, IMAGE_HEIGHT), augment=True)
validation_generator = CustomDataGenerator(val_image_paths, val_labels, BATCH_SIZE, (IMAGE_WIDTH, IMAGE_HEIGHT))

#Building the model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='sigmoid')  # Using sigmoid for multi-label classification
])

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    epochs=EPOCHS
)

# Save the model
model.save('coffee_leaf_disease_classifier.h5')