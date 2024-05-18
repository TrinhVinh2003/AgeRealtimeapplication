import pandas as pd
import numpy as np
import seaborn as sns
import os
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from keras.utils import to_categorical
from preprocessing import x,y_age


from tensorflow.keras.applications import VGG16
from tensorflow.keras import regularizers


x_train_age, x_test_age, y_train_age, y_test_age = train_test_split(x, y_age, test_size=0.2, stratify=y_age,random_state =42)



base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

agemodel = Sequential()

# Add the VGG16 base model to the Sequential model
agemodel.add(base_model)

# Add custom layers for age prediction
agemodel.add(Flatten())
agemodel.add(Dense(500, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
agemodel.add(Dense(200, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
agemodel.add(Dense(9, activation='softmax'))
agemodel.compile(loss='categorical_crossentropy',
             optimizer=optimizers.Adam(lr=0.0001), metrics=['accuracy'])
agemodel.summary()


checkpoint = ModelCheckpoint(filepath=f"age_model_checkpoint.keras",
                             monitor='val_accuracy',
                             save_best_only=True,
                             save_weights_only=False,
                             verbose=1
                            )

tensorboard = TensorBoard("cnn_logs")

datagen = ImageDataGenerator(
      rescale=1./255.,rotation_range=20, width_shift_range = 0.1, height_shift_range = 0.1, horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale=1./255)

trainY = to_categorical(y_train_age, num_classes=7) # [[1, 0], [0, 1], [0, 1], ...]
testY = to_categorical(y_test_age, num_classes=7)
train2 = datagen.flow(x_train_age, trainY, batch_size=64)

test2 = test_datagen.flow(x_test_age, testY, batch_size=64)

history1 = agemodel.fit(train2, epochs=60, shuffle=True, validation_data=test2,callbacks =[checkpoint,tensorboard])
agemodel.save('model_pretrain.h5')

from matplotlib import pyplot as plt
plt.plot(history1.history['accuracy'])
plt.plot(history1.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.legend(['train','test'],loc= 'upper left')
plt.show()

plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'])
plt.title('mdodel loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()