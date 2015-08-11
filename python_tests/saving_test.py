# this is apparently how to save as the MNIST.pkl.gz file format 
# I think it saves in Float64 format (judgin by bn_sne example)

# from here: https://groups.google.com/forum/#!topic/theano-users/7LsADd4bgY0

from numpy import genfromtxt
import gzip, cPickle

train_set_x = genfromtxt('train_x.csv', delimiter=',')
train_set_y = genfromtxt('train_y.csv', delimiter=',')
val_set_x = genfromtxt('val_x.csv', delimiter=',')
val_set_y = genfromtxt('val_y.csv', delimiter=',')
test_set_x = genfromtxt('test_x.csv', delimiter=',')
test_set_y = genfromtxt('test_y.csv', delimiter=',')

train_set = train_set_x, train_set_y
val_set = val_set_x, val_set_y
test_set = test_set_x, val_set_y

dataset = [train_set, val_set, test_set]

f = gzip.open('file.plk.gz','wb')
cPickle.dump(dataset, f, protocol=2)
f.close()