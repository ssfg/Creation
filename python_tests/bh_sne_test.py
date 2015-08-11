# !wget http://deeplearning.net/data/mnist/mnist.pkl.gz

import numpy as np
import gzip, cPickle
import matplotlib.pyplot as plt
import time
import os

# importing the BarnesHut SNE (fast tSNE)
from tsne import bh_sne

from bn_saving import *

# opens and loads the data into different sets (training, validation, test)
f = gzip.open("mnist.pkl.gz", "rb")
train, val, test = cPickle.load(f)
f.close()

# This is the actual matrix data
print "train: ", len(train[0]), len(train[1]), "val: ", len(val[0]), "test: ", len(test[0])

# organise the data into data and labels
X = np.asarray(np.vstack((train[0], val[0], test[0])), dtype=np.float64)
y = np.hstack((train[1], val[1], test[1]))

plot_bn_sne(X,y,800)

print "shape of y: ", y.shape

# print "x-shape before", X.shape
# X = X[0:500] # trims the 70,000 matrix down to 2000 - far faster!
# print "x-shape after", X.shape
# Xdim0 = X.shape[0]
# Xdim1 = X.shape[1]

# # these are the labels
# y = y[0:500] # The more data however, the better the plot


# # this tells it to create the plot?
# X_2d = bh_sne(X)
# print "shape X_2d(0): ", X_2d.shape[0]
# print "shape X_2d(1): ", X_2d.shape[1]

# # rcParams['figure.figsize'] = 20, 20

# # finds the current directory path
# path = os.path.dirname(os.path.realpath(__file__))

# # gets todays date and time
# date_today = time.strftime('%d-%b-%Y')
# time_now = time.strftime('%H-%M-%S')

# # creates a file directory, checks if it exists, if not creates one
# data_directory = "{}/images/{}".format(path, date_today)
# if not os.path.exists(data_directory):
#     os.makedirs(data_directory)

# # create the filename
# filename = 'images/{}/bh_sne-0x{}-1x{}-t{}.png'.format(date_today, Xdim0, Xdim1, time_now)

# # create the scatter plot
# # scatter(x-coordinates?, y-coordinates?, c=(sequence of colours...labels?))
# plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y)
# plt.savefig(filename, dpi=120)
# plt.close()

# # if i%100==0 (i.e - print every certain number of epochs)

