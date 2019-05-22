#!/usr/bin/env python3.7
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

#______________________________ utils import
import os
tokens = os.path.abspath(__file__).split('/')
root_path = '/'.join(tokens[:-2])
print(root_path)


#______________________________ functions
def load_and_format_data(filename):
    dataframe = pandas.read_csv(filename)
    dataset = dataframe.values
    x = dataset[:, 1:].astype(float)
    x = x / 255.0
    y = dataset[:, 0]
    y1 = np_utils.to_categorical(y)
    print(y[0], y1[0])
    x_train, x_test, y_train, y_test = train_test_split(x, y1, test_size=0.33)
    return x_train, y_train, x_test, y_test


#====================================== main
x_train, y_train, x_test, y_test = load_and_format_data('{}/data/train.csv'.format(root_path))
num_pixels = x_train.shape[1]
num_classes = y_train.shape[1]
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
print('num_pixels', num_pixels, 'num_classes', num_classes)

for size in [1000, 2000, 5000]:
    #______________________________________ model
    model = Sequential()
    model.add(Dense(size, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    #______________________________________ train
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=50, verbose=0)

    #______________________________________ evalute model
    scores = model.evaluate(x_test, y_test, verbose=0)
    # print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))
    print('{}: {:0.2f}%'.format(size, 100 - scores[1] * 100))

