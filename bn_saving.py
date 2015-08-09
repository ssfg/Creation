
import numpy as np
import gzip, cPickle
import matplotlib.pyplot as plt
import time
import os

# importing the BarnesHut SNE (fast tSNE)
from tsne import bh_sne


def plot_bn_sne(data, labels, size):

  print "data[0]: ", data.shape
  print "labels[0]: ", labels.shape

  # trim the data & labels down to reasonable size
  data = data[0:size]
  labels = labels[0:size]

  # sizes
  data0 = data.shape[0]
  data1 = data.shape[1]

  # dimensionality reduction with bn_sne
  X_2d = bh_sne(data)
  print "plot shape: ", X_2d.shape

  # plot & save
  plot_save(X_2d, labels, data0, data1)

def plot_save(X_2d, y, Xdim0, Xdim1):

  # finds the current directory path
  path = os.path.dirname(os.path.realpath(__file__))

  # gets todays date and time
  date_today = time.strftime('%d-%b-%Y')
  time_now = time.strftime('%H-%M-%S')

  # creates a file directory, checks if it exists, if not creates one
  data_directory = "{}/images/{}".format(path, date_today)
  if not os.path.exists(data_directory):
      os.makedirs(data_directory)

  # create the filename
  filename = 'images/{}/bh_sne-0x{}-1x{}-t{}.png'.format(date_today, Xdim0, Xdim1, time_now)

  # create the scatter plot
  # scatter(x-coordinates?, y-coordinates?, c=(sequence of colours...labels?))
  plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y)
  plt.savefig(filename, dpi=120)
  plt.close()


