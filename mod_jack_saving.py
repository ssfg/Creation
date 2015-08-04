import gzip
import itertools
import pickle
import os
import sys
import numpy as np
import lasagne
import theano
import theano.tensor as T
import time
import csv


# Imports
from IDAPICourseworkLibrary import *
import h5py

def save_params_slow(filename, epoch, output_layer, ending, out_type):

    print("saving params...")

    # collecting all Tensor Shared Variables [W b W b W b] - weights and biases
    all_params = lasagne.layers.get_all_params(output_layer)
    # collecting a List of Numpy Arrays - all weights and biases
    param_values = lasagne.layers.get_all_param_values(output_layer)
    # another way to do this
    # all_param_values = [p.get_value() for p in all_params]

    # checking sizes
    no_arrays = len(all_params)
    print ("all params: ", all_params)
    print ("no arrays: ", no_arrays)

    # Going through, and saving each matrix to a separate file (not printing all, too slow??)
    for i, data in enumerate(param_values):
        layer = 1
        w_or_b = "na"

        # odd ones are weights, indexed from one?
        if (i % 2) != 0:
            w_or_b = "bias"
            layer += 1
        else:
            w_or_b = "weights"

        # saving the filename: epoch, layer, w/b
        filename_unique = "{}-E{}-L{}-{}.{}".format(filename, epoch['number'], layer, w_or_b, ending)

        print filename_unique
        # print data

        # different methods to output data - not all work :|
        if(out_type=="CSV"):
            with open(filename, "wb") as f:
                writer = csv.writer(f)
                writer.writerows(data)
        elif(out_type=="PICKLE"):
            with open(filename, 'w') as f:
                pickle.dump(data, f)
        elif(out_type=="NUMPY"):
            numpy.savetxt(filename_unique, data, delimiter=",")
        elif(out_type=="JSON"):
            print "yet to be implemented"

    print("params saved!") # hurray


def save_activations_test(filename, epoch, dataset, output_layer, ending, out_type):

    print ("Saving Activations...")

    th_layers = lasagne.layers.get_all_layers(output_layer)

    X_val = dataset['X_valid']
    # print X_val.eval()

    for i, layer in enumerate(th_layers):
        # only care about layers with params
        if not (layer.get_params() or isinstance(layer, lasagne.layers.FeaturePoolLayer)):
            continue

        data = lasagne.layers.get_output(layer, X_val, deterministic=True).eval()

        filename_unique = "{}-E{}-L{}.{}".format(filename, epoch['number'], i, ending)

        print filename_unique

        # different methods to output data - not all work :|
        if(out_type=="CSV"):
            with open(filename, "wb") as f:
                writer = csv.writer(f)
                writer.writerows(data)
        elif(out_type=="PICKLE"):
            with open(filename, 'w') as f:
                pickle.dump(data, f)
        elif(out_type=="NUMPY"):
            numpy.savetxt(filename_unique, data, delimiter=",")
        elif(out_type=="JSON"):
            print "yet to be implemented"
        
    print("activations saved!")

        # n_features = output.shape[-1]
        # seq_length = int(output.shape[0] / )

        # if isinstance(layer, DenseLayer):
        #     shape = 
        #     output = output.reshape(shape)
        # elif isinstance(layer, Conv1DLayer):
        #     output = output.transpose(0,2,1)

        # if epoch['number'] == 0:
        #     f.create_dataset('validation_data', data=X_val)



def save_activations_test2(filename, epoch, output_layer):

    print ("Saving Activations...")

    th_layers = lasagne.layers.get_all_layers(output_layer)

    for i, layer in enumerate(th_layers):

        output_act = lasagne.layers.get_output(layer).eval()
        print(output_act)



def save_activations(self):
    if not self.do_save_activations:
        return
    filename = self.experiment_name + "_activations.hdf5"
    mode = 'w' if self.n_iterations() == 0 else 'a'
    f = h5py.File(filename, mode=mode)
    epoch_name = 'epoch{:06d}'.format(self.n_iterations())
    try:
        epoch_group = f.create_group(epoch_name)
    except ValueError:
        self.logger.exception("Cannot save params!")
        f.close()
        return

    layers = get_all_layers(self.layers[-1])

    for layer_i, layer in enumerate(layers):
        # We only care about layers with params
        if not (layer.get_params() or isinstance(layer, FeaturePoolLayer)):
            continue

        output = lasagne.layers.get_output(layer, self.X_val).eval()
        n_features = output.shape[-1]
        seq_length = int(output.shape[0] / self.source.n_seq_per_batch)

        if isinstance(layer, DenseLayer):
            shape = (self.source.n_seq_per_batch, seq_length, n_features)
            output = output.reshape(shape)
        elif isinstance(layer, Conv1DLayer):
            output = output.transpose(0, 2, 1)

        layer_name = 'L{:02d}_{}'.format(layer_i, layer.__class__.__name__)
        epoch_group.create_dataset(
            layer_name, data=output, compression="gzip")

    # save validation data
    if self.n_iterations() == 0:
        f.create_dataset(
            'validation_data', data=self.X_val, compression="gzip")

    f.close()

