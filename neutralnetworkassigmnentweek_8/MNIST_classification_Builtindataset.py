# MNist classification dataset using CNN
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix

np.random.seed(0)
# Data loads 
from keras.datasets import mnist
(x_train, y_train),(x_test, y_test)= mnist.load_data()
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# visualizations
num_classes = 10
f, ax = plt.subplots(1, num_classes, figsize=(20, 20))

for i in range (0, num_classes, ):
    sample = x_train[y_train==i][0]
    ax[i].imshow(sample, cmap = 'gray')
    ax[i].set_title("Label: {}".format(i),fontsize = 16)

for i in range (10):
    print(y_train[i])

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
for i in range (10):
    print(y_train[i])

# prepare data 
#normalization
x_train = x_train / 255.0
x_test = x_test / 255.0

# reshape data
x_train =x_train.reshape(x_train.shape[0],-1)
x_test = x_test.reshape(x_test.shape[0],-1)
print(x_train.shape)

# create a model of CNN 
model = Sequential()
model.add(Dense(units=128, input_shape=(784,), activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(units=10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# traning data
batch_size = 512
epochs = 10
model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs)

#Evaluate data
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test_loss: {}, Test Accuracy: {}".format(test_loss, test_acc))

y_pred =model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
print(y_pred)
print(y_pred_classes)

random_idx = np.random.choice(len(x_test))
x_sample = x_test[random_idx]
y_true = np.argmax(y_test, axis=1)
y_sample_true = y_true[random_idx]
y_sample_pred_classes = y_pred_classes[random_idx]

plt.title("Predicted: {}, True: {}".format(y_sample_pred_classes, y_sample_true, fontsize = 16))
plt.imshow(x_sample.reshape(28,28), cmap='gray')
plt.show()
read = input("waite...")

# confusion matrix
confusion_mtx = confusion_matrix(y_true, y_pred_classes)
fig, ax = plt.subplots(figsize = (15,10))
ax = sns.heatmap(confusion_mtx,annot=True, fmt='d', ax=ax, cmap='Blues')
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
ax.set_title('Confusion Matrix');
read = input("waite...")

