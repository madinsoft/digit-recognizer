# import numpy
import pandas
import matplotlib.pyplot as plt
#______________________________ utils import
import os
tokens = os.path.abspath(__file__).split('/')
root_path = '/'.join(tokens[:-2])
print(root_path)

dataframe = pandas.read_csv('{}/data/train.csv'.format(root_path))
dataset = dataframe.values
X = dataset[:, 1:].astype(int)
XB = X.reshape((len(X), 28, 28))
print(len(X), X)
plt.subplot(221)
plt.imshow(XB[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(XB[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(XB[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(XB[3], cmap=plt.get_cmap('gray'))
plt.show()
