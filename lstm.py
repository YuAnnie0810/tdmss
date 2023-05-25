import keras.callbacks
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
import numpy as np
import os
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

batch_size = 256
epochs = 1000
log_dir = os.path.join('Logs_1hand')
tb_callback = TensorBoard(log_dir=log_dir)
stop_callback = keras.callbacks.EarlyStopping(patience=10, monitor='val_categorical_accuracy', min_delta=0.0005)


def build_and_train_model_2hands(X_train, y_train, actions):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,126)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(33, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.fit(X_train, y_train, epochs=epochs, shuffle=True, batch_size=batch_size,
              validation_split=0.05, callbacks=[tb_callback, stop_callback])
    return model


def build_and_train_model_1hand(X_train, y_train, actions):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 63)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    #model.add(Dense(33, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.fit(X_train, y_train, epochs=epochs, shuffle=True, batch_size=128,
              validation_split=0.05, callbacks=[tb_callback, stop_callback])
    return model


def accuracy_score_find(model, X_test, y_test):
    y_hat = model.predict(X_test)
    y_true = np.argmax(y_test, axis=1).tolist()
    y_hat = np.argmax(y_hat, axis=1).tolist()
    #multilabel_confusion_matrix(y_true, y_hat)
    return accuracy_score(y_true, y_hat)