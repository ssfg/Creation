import numpy as np
import matplotlib.pyplot as plt
import base64
import pandas as pd
import os
import sys
import glob

# new_data = np.genfromtxt('test_csv.csv', dtype=np.float32, delimiter=',')


def find_files():
  path = os.path.dirname(os.path.realpath(__file__))
  folder = path + "/csv_colah"

  # glob.glob alows therminal stype collecting of data, i.e the *.csv
  allFiles = glob.glob(folder + "/*.csv")

  # print glob.glob(path)

  print folder

  return allFiles

  # big_array = [] #  empty regular list
  #   for i in range(5):
  #       arr = i*np.ones((2,4)) # for instance
  #       big_array.append(arr)
  #   big_np_array = np.array(big_array)  # transformed to a numpy array

def method_one_concatenation():

  allFiles = find_files()

  # initiates by getting first file
  intitial = np.genfromtxt(allFiles[0], delimiter=",")
  # trims the NaNs from the end (the '\n's)
  intitial = intitial[:,:-1]
  # concatenates all rows 
  intitial = np.reshape(intitial, (1,-1))

  count = 0

  # goes through all the files in the list of files saved
  for i, file in enumerate(allFiles):
    # to not append to the one we initialised with above 
    if i != 0:
      # gets the contents
      df = np.genfromtxt(file, delimiter=",")
      # trims the NaN
      df = df[:,:-1]
      # concatenates all rows
      df = np.reshape(df, (1,-1))
      # appends to first one (need to all be the same shape)
      intitial = np.append(intitial, df, axis=0)
      count = count  + 1
      print count

  
  # trim to make reasonable
  # intitial = intitial[0:500]

  print "shape: ", intitial.shape
  np.savetxt("test.csv", intitial, delimiter=",")

def method_two_nested():

  allFiles = find_files()

  # initiates by getting the first file
  intitial = np.genfromtxt(allFiles[0], delimiter=",")
  # trims the NaNs from the end (the '\n's)
  intitial = intitial[:,:-1]

  count = 0

  # goes through all the files in the list of files saved
  for i, file in enumerate(allFiles):
    # to not append to the one we initialised with above 
    if i != 0:
      # gets the contents
      df = np.genfromtxt(file, delimiter=",")
      # trims the NaN
      df = df[:,:-1]
      # appends to first one (need to all be the same shape)
      intitial = np.append(intitial, df, axis=0)
      count = count  + 1
      print count

  print "shape: ", intitial.shape

  np.savetxt("test2.csv", intitial, delimiter=",")

# The Pandas method found online
def pandas():

  path =r"C:\DRO\DCL_rawdata_files"
  allFiles = glob.glob(path + "/*.csv")
  frame = pd.DataFrame()
  list = []
  for file in allFiles:
      df = pd.read_csv(str.join(path,file),index_col=None, header=0)
      list.append(df)
  frame = pd.concat(list)

if __name__ == "__main__":
  method_one_concatenation(); # shape: 60, 20000
  # method_two_nested();      # shape: 600000, 2
