
# """"
# maize disease classifier model
# """
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# import json

# from iGrow.src.teaDisease_classifier import BATCH_SIZE, IMAGE_HEIGHT

# #Defining the constants
# IMAGE_WIDTH, IMAGE_HEIGHT = 150
# BATCH_SIZE = 32
# EPOCHS = 10
# DATA_DIR = r"E:/A2SV Hackathon/Maize disease dataset/data"

# #Data generators
# datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)

# train_generator = datagen.flow_from_directory(
#     DATA_DIR,
#     target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
#     batch_size=BATCH_SIZE,
#     class_mode='categorical',
#     subset='training'
# )

# validation_generator = datagen.flow_from_directory(
#     DATA_DIR,
#     target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
#     batch_size=BATCH_SIZE,
#     class_mode='categorical',
#     subset='validation'
# )

# #Building the model
# model = Sequential([
#     Conv2D(32, (3,3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)),
#     MaxPooling2D(2,2),
#     Conv2D(64, (3,3), activation='relu'),
#     MaxPooling2D(2,2),
#     Conv2D(128, (3,3), activation='relu'),
#     MaxPooling2D(2,2),
#     Flatten(),
#     Dense(512, activation='relu'),
#     Dropout(0.5),
#     Dense(train_generator.num_classes, activation='softmax')
#     #Dense(4, activation='softmax')
# ])

# #compile the model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# #Training the model
# model.fit(
#     train_generator,
#     epochs= EPOCHS,
#     validation_data = validation_generator
# )

# #saving the model
# model.save('maize_disease_classifier.h5')

# # Save the class indices
# class_indices = train_generator.class_indices
# with open('maize_disease_class_indices.json', 'w') as f:
#     json.dump(class_indices, f)


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import os

#Define the constants
IMAGE_WIDTH = 150
IMAGE_HEIGHT = 150
BATCH_SIZE = 32
EPOCHS = 15
NUM_CLASSES = 8

#Loading and preprocessing the data
data_dir = r"E:/A2SV Hackathon/tea dataset Kaggle/tea sickness dataset"
datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size = BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size = BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)
#printing the class indices to verify the mapping
print(train_generator.class_indices)

#Building the model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Dense(512, activation='relu'),
    Dense(NUM_CLASSES, activation='softmax')
])

#compiling the model
model.compile(optimizer=Adam(), 
            loss='categorical_crossentropy', 
            metrics=['accuracy'])

#Training the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=EPOCHS
)

#saving the model
model.save('maize_disease_classifier.h5')