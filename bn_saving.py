
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

# pca and lda imports
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.lda import LDA

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
  X_2d = bh_sne(data, perplexity=19.0, theta=0.5)
  print "plot shape: ", X_2d.shape

  # plot & save
  plot_save(X_2d, labels, data0, data1, "bn-SNE")

def plot_pca(data,labels,size):
  
  print "data[0]: ", data.shape
  print "labels[0]: ", labels.shape

  # trim the data & labels down to reasonable size
  data = data[0:size]
  labels = labels[0:size]

  # sizes
  data0 = data.shape[0]
  data1 = data.shape[1]

  pca = PCA(n_components=data1)
  X_r = pca.fit(data).transform(data)

  # dimensionality reduction with bn_sne
  # X_2d = bh_sne(data, perplexity=19.0, theta=0.5)

  print "plot shape: ", X_r.shape

  # plot & save
  plot_save(X_r, labels, data0, data1, "pca")


def plot_save(X_2d, y, Xdim0, Xdim1, Dtype):

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
  filename = 'images/{}/{}-0x{}-1x{}-t{}.png'.format(date_today, Dtype, Xdim0, Xdim1, time_now)

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

    plot_save(X_2d, label, Xdim0, Xdim1, "coord")

def read_and_plot_2():
    my_data = np.genfromtxt('./PCA-SNE.csv', delimiter=",")
    my_labels = np.genfromtxt('./PCA_SNE_labels_index.csv', delimiter="\n")

    # keeping the data in 2D format
    X_2d = my_data[:1000]
    label = my_labels[:1000]
    Xdim0 = X_2d.shape[0]
    Xdim1 = X_2d.shape[1]

    print "shapes: ", Xdim0, ", ", Xdim1

    plot_save(X_2d, label, Xdim0, Xdim1, "coord")

def read_sne_plot():
    my_data = np.genfromtxt('./test_data.csv', delimiter=',')
    labels = np.genfromtxt('./test_labels.csv', delimiter=',')

    print "data incoming shape", my_data.shape
    # getting X, y and labels - also trims the NaNs

    # labels = my_labels[:,0]

    # keeping the data in 2D format
    # should trim the third column
    X_2d = my_data[:,:-1]
    
    plot_bn_sne(X_2d, labels, 1000)

def read_pca_plot():
    my_data = np.genfromtxt('./test_data.csv', delimiter=',')
    labels = np.genfromtxt('./test_labels.csv', delimiter=',')

    print "data incoming shape", my_data.shape
    # getting X, y and labels - also trims the NaNs

    # labels = my_labels[:,0]

    # keeping the data in 2D format
    # should trim the third column
    X_2d = my_data[:,:-1]
    
    plot_pca(X_2d, labels, 10000)

def read_sne_video():
    my_data = np.genfromtxt('./test_data.csv', delimiter=',')
    labels = np.genfromtxt('./test_labels.csv', delimiter=',')

    print "data incoming shape", my_data.shape
    # getting X, y and labels - also trims the NaNs

    # labels = my_labels[:,0]

    # keeping the data in 2D format
    # should trim the third column
    data = my_data[:,:-1]

    X_2d = bh_sne(data, perplexity=19.0, theta=0.5)

    makeVideo(X_2d, labels)
    
    
def makeVideo(X_2d,labels):

    # doesn't work atm: gives error (AttributeError: 'FigureCanvasMac' object has no attribute ....
    name = "test.gif"
    fps = 10
    duration = X_2d.shape[0]

    # gets the figure (here 1) and the axis array (ax1, ax2)
    fig, ax = plt.subplots(1, figsize=(4, 4), facecolor='white')

    def make_frame(t):
        ax.clear()
        print(t)

        ax.set_title("Activations", fontsize=16)
        ax.scatter(X_2d[:,0],X_2d[:,1],alpha=0.1,lw=0.0)

        ax.scatter(X_2d[t*fps:t*fps+1,0],X_2d[t*fps:t*fps+1,1],alpha=1,lw=0.0)

        return mplfig_to_npimage(fig)

    animation = mpy.VideoClip(make_frame, duration = duration)
    animation.write_gif2(name, fps=gif_fps)


if __name__ == "__main__":


    # read_and_plot_2() # i.e - the meta-SNE version

    # read_sne_plot()
    # read_pca_plot()
    read_sne_video()