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

#lấy dữ liệu trong tệp ảnh UTKFACE-new
images = []
ages = []
for i in os.listdir('../input/utkface-new/UTKFace/')[0:23600]:
    split = i.split('_')
    ages.append(int(split[0]))
    images.append(Image.open('../input/utkface-new/UTKFace/' + i))
for i in os.listdir('../input/utkface-new/crop_part1/')[0:9000]:
    split = i.split('_')
    ages.append(int(split[0]))
    images.append(Image.open('../input/utkface-new/crop_part1/' + i))


# Gán nhãn theo khoảng độ tuổi ví dụ (22-25) =>3
def class_labels_reassign(age):
    if 1 <= age <= 2:
        return 0
    elif 4 <= age <= 13:
        return 1
    elif 14 <= age <= 21:
        return 2
    elif 22 <= age <= 25:
        return 3
    elif 26 <= age <= 28:
        return 4
    elif 29 <= age <= 34:
        return 5
    elif 37 <= age <= 48:
        return 6
    elif 49 <= age <= 64:
        return 7
    else:
        return 8

    df['Ages'] = df['Ages'].map(class_labels_reassign)


# ta thấy dữ liệu từ nhãn 4 trở lên có sự chênh lệch ảnh khá lớn , vì vậy ta sẽ giảm bớt đi để dữ liệu cân bằng
filtered_df = df[df['Ages'] >= 4]

under4s = filtered_df.sample(frac=0.7, random_state=42)

df = df[df['Ages'] < 4]

df = pd.concat([df, under4s], ignore_index=True)


# resize ảnh về dạng 128x 128 và đưa về dữ liệu số
x = []
for i in range(len(df)):
    df['Images'].iloc[i] = df['Images'].iloc[i].resize((128, 128), Image.ANTIALIAS)
    ar = np.asarray(df['Images'].iloc[i])
    x.append(ar)

x = np.array(x)

y_age = df['Ages']