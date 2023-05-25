from keras.models import load_model
from keras.utils import to_categorical
import numpy as np
import os
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import train_test_split
from data_collection import actions, no_sequences, DATA_PATH, sequence_length

model = load_model('rsl_model4.h5')

label_map = {label: num for num, label in enumerate(actions)}
print(label_map)

sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        print(str(action) + ' ' + str(sequence))
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

print('collected data')

X = np.array(sequences)
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

y_hat = model.predict(X_test)
y_true = np.argmax(y_test, axis=1).tolist()
y_hat = np.argmax(y_hat, axis=1).tolist()
print(multilabel_confusion_matrix(y_true, y_hat))