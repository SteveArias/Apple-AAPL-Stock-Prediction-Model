import numpy as np
import pandas as pd
import time
import warnings
import keras
import matplotlib.pyplot as plt
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers import Flatten
from keras.models import Sequential
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

np.random.seed(0)
data = pd.read_csv("data.csv")

y = data['Future Close (AAPL)']
X = data.drop('Future Close (AAPL)', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, shuffle=False)

print("Size of X_train: ")
print(X_train.shape)
print("\nSize of y_train: ")
print(y_train.shape)
print("Size of X_test: ")
print(X_test.shape)
print("\nSize of y_test: ")
print(y_test.shape)  

X_train_feature_scaled = (X_train - X_train.mean()) / X_train.std()
y_train_feature_scaled = (y_train - y_train.mean()) / y_train.std()

X_test_feature_scaled = (X_test - X_test.mean()) / X_test.std()
y_test_feature_scaled = (y_test - y_test.mean()) / y_test.std()

model = Sequential()
X_train = X_train_feature_scaled.values
X_test = X_test_feature_scaled.values

y_test = y_test.values
y_train = y_train.values

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

model.add(LSTM(input_shape=(53, 1), output_dim=1, return_sequences=True));
model.add(Dropout(0.2))

model.add(LSTM(units=100,return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(output_dim=1))
model.add(Activation('linear'))

start = time.time()
model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
print ('compilation time : ' + str(time.time() - start))
model.summary();


history = model.fit(X_train, y_train, batch_size=500, epochs=200, validation_split=0.1)
results = model.predict(X_test)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

plt.scatter(range(640), results, c='r')
plt.scatter(range(640), y_test, c='g')
plt.show()

#predictions = lstm.predict_sequences_multiple(model, X_test, 50, 50)
#lstm.plot_results_multiple(predictions, y_test, 50)



