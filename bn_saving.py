
import numpy as np
import gzip, cPickle
import matplotlib.pyplot as plt
import time
import os

# used for getting all files in a folder
import pandas as pd
import sys
import glob

# importing the BarnesHut SNE (fast tSNE)
from tsne import bh_sne

# We'll generate an animation with matplotlib and moviepy.
from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy



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


def read_and_plot():
    my_data = np.genfromtxt('./csv_colah/mnist_conv5net2_N0.csv', delimiter=',')
    my_labels = np.genfromtxt('./csv_txt_tests/mnist_ys.csv', delimiter=',')

    print "data incoming shape", my_data.shape
    # getting X, y and labels - also trims the NaNs
    X = my_data[:1000,0]
    y = my_data[:1000,1]
    label = my_labels[:1000,0]

    # keeping the data in 2D format
    X_2d = my_data[:1000]
    Xdim0 = X_2d.shape[0]
    Xdim1 = X_2d.shape[1]

    print "shapes: ", Xdim0, ", ", Xdim1

    plot_save(X_2d, label, Xdim0, Xdim1)

def read_and_plot_2():
    my_data = np.genfromtxt('./PCA-SNE.csv', delimiter=",")
    my_labels = np.genfromtxt('./PCA_SNE_labels_index.csv', delimiter="\n")

    # keeping the data in 2D format
    X_2d = my_data[:1000]
    label = my_labels[:1000]
    Xdim0 = X_2d.shape[0]
    Xdim1 = X_2d.shape[1]

    print "shapes: ", Xdim0, ", ", Xdim1

    plot_save(X_2d, label, Xdim0, Xdim1)



def read_sne_plot():
    my_data = np.genfromtxt('./csv_colah/mnist_conv5net2_N0.csv', delimiter=',')
    my_labels = np.genfromtxt('./csv_txt_tests/mnist_ys.csv', delimiter=',')

    print "data incoming shape", my_data.shape
    # getting X, y and labels - also trims the NaNs
    X = my_data[:1000,0]
    y = my_data[:1000,1]
    label = my_labels[:1000,0]

    # keeping the data in 2D format
    X_2d = my_data[:1000]
    Xdim0 = X_2d.shape[0]
    Xdim1 = X_2d.shape[1]

    print "shapes: ", Xdim0, ", ", Xdim1

    plot_save(X_2d, label, Xdim0, Xdim1)


if __name__ == "__main__":


    read_and_plot_2() # i.e - the meta-SNE version
