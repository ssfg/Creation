def save(self):
        self.logger.info("Saving plots...")
        try:
            self.plotter.plot_all()
        except:
            self.logger.exception("")
        self.logger.info("Saving params...")
        try:
            self.save_params()
        except:
            self.logger.exception("")
        self.logger.info("Saving activations...")
        try:
            self.save_activations()
        except:
            self.logger.exception("")
        self.logger.info("Finished saving.")

    def n_iterations(self):
        return max(len(self.training_costs) - 1, 0)

    def save_params(self, filename=None):
        """
        Save it to HDF in the following format:
            /epoch<N>/L<I>_<type>/P<I>_<name>
        """
        if filename is None:
            filename = self.experiment_name + ".hdf5"

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
            params = layer.get_params()
            if not params:
                continue
            layer_name = 'L{:02d}_{}'.format(layer_i, layer.__class__.__name__)
            layer_group = epoch_group.create_group(layer_name)
            for param_i, param in enumerate(params):
                param_name = 'P{:02d}'.format(param_i)
                if param.name:
                    param_name += "_" + param.name
                data = param.get_value()
                layer_group.create_dataset(
                    param_name, data=data, compression="gzip")

        f.close()

    def load_params(self, iteration, path=None):
        """
        Load params from HDF in the following format:
            /epoch<N>/L<I>_<type>/P<I>_<name>
        """
        # Process function parameters
        filename = self.experiment_name + ".hdf5"
        if path is not None:
            filename = join(path, filename)
        self.logger.info('Loading params from ' + filename + '...')

        f = h5py.File(filename, mode='r')
        epoch_name = 'epoch{:06d}'.format(iteration)
        epoch_group = f[epoch_name]

        layers = get_all_layers(self.layers[-1])
        for layer_i, layer in enumerate(layers):
            params = layer.get_params()
            if not params:
                continue
            layer_name = 'L{:02d}_{}'.format(layer_i, layer.__class__.__name__)
            layer_group = epoch_group[layer_name]
            for param_i, param in enumerate(params):
                param_name = 'P{:02d}'.format(param_i)
                if param.name:
                    param_name += "_" + param.name
                data = layer_group[param_name]
                param.set_value(data.value)
        f.close()
        self.logger.info('Done loading params from ' + filename + '.')

        # LOAD COSTS
        def load_csv(key, limit):
            filename = self.csv_filenames[key]
            if path is not None:
                filename = join(path, filename)
            data = np.genfromtxt(filename, delimiter=',', skip_header=1)
            data = data[:limit, :]

            # overwrite costs file
            self._write_csv_headers(key)
            with open(filename, mode='a') as fh:
                np.savetxt(fh, data, delimiter=',')
            return list(data[:, 1])

        self.training_costs = load_csv('training_costs', iteration)
        self.validation_costs = load_csv(
            'validation_costs', iteration // self.validation_interval)

        # LOAD TRAINING COSTS METADATA
        metadata_fname = self.csv_filenames['training_costs_metadata']
        if path is not None:
            metadata_fname = join(path, metadata_fname)
        try:
            metadata_fh = open(metadata_fname, 'r')
        except IOError:
            pass
        else:
            reader = csv.DictReader(metadata_fh)
            training_costs_metadata = [row for row in reader]
            keys = training_costs_metadata[-1].keys()
            metadata_fh.close()
            self.training_costs_metadata = training_costs_metadata[:iteration]
            if len(training_costs_metadata) > iteration:
                # Overwrite old file
                with open(metadata_fname, 'w') as metadata_fh:
                    writer = csv.DictWriter(metadata_fh, keys)
                    writer.writeheader()
                    writer.writerows(self.training_costs_metadata)

        # set learning rate
        if self.learning_rate_changes_by_iteration:
            keys = self.learning_rate_changes_by_iteration.keys()
            keys.sort(reverse=True)
            for key in keys:
                if key < iteration:
                    self.learning_rate = (
                        self.learning_rate_changes_by_iteration[key])
                    break

        # epoch_callbacks
        callbacks_to_call = [
            key for key in self.epoch_callbacks.keys() if key < iteration]
        for callback_iteration in callbacks_to_call:
            self.epoch_callbacks[callback_iteration](self, callback_iteration)

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


def _write_csv_row(filename, row, mode='a'):
    with open(filename, mode=mode) as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(row)

