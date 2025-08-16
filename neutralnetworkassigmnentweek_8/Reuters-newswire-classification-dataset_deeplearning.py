# Reuters newswire classification dataset apply cnn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.datasets import reuters
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout


num_words = 10000
maxlen = 200

(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=num_words)
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

num_classes = np.max(y_train) + 1
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

model = Sequential()
model.add(Embedding(num_words, 128, input_length=maxlen))
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
history = model.fit(x_train, y_train_cat,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2,
                    verbose=1)

loss, accuracy = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"Test Accuracy: {accuracy:.4f}")

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = y_test  

cm = confusion_matrix(y_true, y_pred_classes)

# Plot confusion matrix heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(cm, cmap="Blues", annot=False, fmt="d")
plt.title("Confusion Matrix - Reuters Newswire Classification")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()