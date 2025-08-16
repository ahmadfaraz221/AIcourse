# Cifar100 small image classification through neural networking 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keras import Sequential, datasets
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten
from sklearn.metrics import confusion_matrix

# load dataset
(x_train, y_train),(x_test, y_test) = datasets.cifar100.load_data()

# reshape data 2D array into 1D array
y_train = y_train.reshape(-1,)
y_test = y_test.reshape(-1,)

# create a plot for visualize 
plt.figure(figsize=(10,10))
for image in range(0,20):
    i = image
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    j=i+0
    data_plot = x_train[j]
    plt.imshow(data_plot)
    plt.xlabel(str(y_test[j]))
plt.show()

# normalize the training data and test data set
x_train = x_train/255
x_test = x_test/255

model = Sequential()
model.add(Conv2D(input_shape=(32,32,3), kernel_size=(2,2), padding='same', strides=(2,2), filters=32))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same'))
model.add(Conv2D(kernel_size=(2,2), padding='same', strides=(2,2), filters=64))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(100, activation='softmax'))
model.summary()

opt = 'adam'
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=25)
test_loss, test_acc = model.evaluate(x_train, y_train)
print("test accuracy: ", test_acc)


def plot_comfusion_matrix(cm, classes, normalize = True,
                          title = 'confusion metrix',
                          cmap = plt.cm.Reds):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if normalize:
        cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
    plt.tight_layout()
    plt.ylabel('observation')
    plt.xlabel('pridiction')
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
confusion_mtx = confusion_matrix(y_test, y_pred_classes)
plot_comfusion_matrix(confusion_mtx, classes=range(100))
plt.show()


