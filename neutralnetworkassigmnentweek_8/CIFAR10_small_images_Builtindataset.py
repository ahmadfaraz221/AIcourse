#CIFAR10 small images builtin data set classifiaction
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, datasets
from sklearn.metrics import confusion_matrix, classification_report
from keras import datasets, layers, models

# dataset
from keras.datasets import cifar10
(x_train, y_train),(x_test, y_test)= cifar10.load_data()
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# visulization
y_train = y_train.flatten()
y_test = y_test.flatten()

# CIFAR-10 classes
class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

# Plot one sample per class
fig, ax = plt.subplots(1, 10, figsize=(20, 3))
for i in range(10):
    sample = x_train[y_train == i][0]   # first image of class i
    ax[i].imshow(sample)
    ax[i].set_title(class_names[i])
    ax[i].axis("off")

plt.show()


x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

# Build CNN model
cnn = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

cnn.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

cnn.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
cnn.evaluate(x_test, y_test)

y_test = y_test.reshape(-1,)
y_test[:5]

y_pred = cnn.predict(x_test)
y_pred[:5]
y_classes = [np.argmax(element) for element in y_pred]
y_classes[:5]


print("classification report: /n", classification_report(y_test, y_classes))

y_true = y_true = y_test.flatten()
y_pred_classes = np.argmax(y_pred, axis=1)
confusion_mtx = confusion_matrix(y_true, y_pred_classes)
fig, ax = plt.subplots(figsize = (15,10))
ax = sns.heatmap(confusion_mtx,annot=True, fmt='d', ax=ax, cmap='Blues')
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
ax.set_title('Confusion Matrix');
read = input("waite...")
