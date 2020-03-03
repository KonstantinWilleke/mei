import pickle

import torch

from nnfabrik.utility.nn_helpers import get_dims_for_loader_dict
from nnfabrik.utility.nnf_helper import split_module_name, dynamic_import
from nnfabrik.utility.dj_helpers import make_hash


def get_output_selected_model(neuron_pos, session_id, model):
    """Creates a version of the model that has its output selected down to a single uniquely identified neuron.

    Args:
        neuron_pos: An integer, the position of the neuron in the model's output.
        session_id: A string that uniquely identifies one of the model's readouts.
        model: A PyTorch module that can be called with a keyword argument called "data_key". The output of the
            module is expected to be a two dimensional Torch tensor where the first dimension corresponds to the
            batch size and the second to the number of neurons.

    Returns:
        A function that takes the model input(s) as parameter(s) and returns the model output corresponding to the
        selected neuron.
    """

    def output_selected_model(x, *args, **kwargs):
        output = model(x, *args, data_key=session_id, **kwargs)
        return output[:, neuron_pos]

    return output_selected_model


def get_mappings(dataset_config, key, load_func=None):
    if load_func is None:
        load_func = load_pickled_data
    entities = []
    for datafile_path in dataset_config["datafiles"]:
        data = load_func(datafile_path)
        for neuron_pos, neuron_id in enumerate(data["unit_indices"]):
            entities.append(dict(key, neuron_id=neuron_id, neuron_position=neuron_pos, session_id=data["session_id"]))
    return entities


def load_pickled_data(path):
    with open(path, "rb") as datafile:
        data = pickle.load(datafile)
    return data


def get_input_shape(dataloaders, get_dims_func=get_dims_for_loader_dict):
    """Gets the shape of the input that the model expects from the dataloaders."""
    return list(get_dims_func(dataloaders["train"]).values())[0]["inputs"]


def import_module(path):
    return dynamic_import(*split_module_name(path))


class ModelLoader:
    def __init__(self, model_table, cache_size_limit=10):
        self.model_table = model_table
        self.cache_size_limit = cache_size_limit
        self.cache = dict()

    def load(self, key):
        if self.cache_size_limit == 0:
            return self._load_model(key)
        if not self._is_cached(key):
            self._cache_model(key)
        return self._get_cached_model(key)

    def _load_model(self, key):
        return self.model_table().load_model(key=key)

    def _is_cached(self, key):
        if self._hash_trained_model_key(key) in self.cache:
            return True
        return False

    def _cache_model(self, key):
        """Caches a model and makes sure the cache is not bigger than the specified limit."""
        self.cache[self._hash_trained_model_key(key)] = self._load_model(key)
        if len(self.cache) > self.cache_size_limit:
            del self.cache[list(self.cache)[0]]

    def _get_cached_model(self, key):
        return self.cache[self._hash_trained_model_key(key)]

    def _hash_trained_model_key(self, key):
        """Creates a hash from the part of the key corresponding to the primary key of the trained model table."""
        return make_hash({k: key[k] for k in self.model_table().primary_key})


def hash_list_of_dictionaries(list_of_dicts):
    """Creates a hash from a list of dictionaries that uniquely identifies the provided list of dictionaries.

    The keys of every dictionary in the list and the list itself are sorted before creating the hash.

    Args:
        list_of_dicts: List of dictionaries.

    Returns:
        A string representing the hash that uniquely identifies the provided list of dictionaries.
    """
    dict_of_dicts = {make_hash(d): d for d in list_of_dicts}
    sorted_list_of_dicts = [dict_of_dicts[h] for h in sorted(dict_of_dicts)]
    return make_hash(sorted_list_of_dicts)


class EnsembleModel:
    """A ensemble model consisting of several individual ensemble members.

    Attributes:
        *members: PyTorch modules representing the members of the ensemble.
    """

    def __init__(self, *members):
        """Initializes EnsembleModel."""
        self.members = members

    def __call__(self, x, *args, **kwargs):
        """Calculates the forward pass through the ensemble.

        The input is passed through all individual members of the ensemble and their outputs are averaged.

        Args:
            x: A tensor representing the input to the ensemble.
            *args: Additional arguments will be passed to all ensemble members.
            **kwargs: Additional keyword arguments will be passed to all ensemble members.

        Returns:
            A tensor representing the ensemble's output.
        """
        outputs = [m(x, *args, **kwargs) for m in self.members]
        mean_output = torch.stack(outputs, dim=0).mean(dim=0)
        return mean_output

    def eval(self):
        """Switches all ensemble members to evaluation mode."""
        for member in self.members:
            member.eval()

    def cuda(self):
        """Transfers the parameters of all ensemble members a CUDA device."""
        for member in self.members:
            member.cuda()


class ConstrainedOutputModel:
    """A model that has its output constrained.

    Attributes:
        model: A PyTorch module.
        constraint: An integer representing the index of a neuron in the model's output. Only the value corresponding
            to that index will be returned.
        forward_kwargs: A dictionary containing keyword arguments that will be passed to the model every time it is
            called. Optional.
    """

    def __init__(self, model, constraint, forward_kwargs=None):
        """Initializes ConstrainedOutputModel."""
        self.model = model
        self.constraint = constraint
        self.forward_kwargs = forward_kwargs if forward_kwargs else dict()

    def __call__(self, x, *args, **kwargs):
        """Computes the constrained output of the model.

        Args:
            x: A tensor representing the input to the model.
            *args: Additional arguments will be passed to the model.
            **kwargs: Additional keyword arguments will be passed to the model.

        Returns:
            A tensor representing the constrained output of the model.
        """
        output = self.model(x, *args, **self.forward_kwargs, **kwargs)
        return output[self.constraint]

    def eval(self):
        """Switches the model to evaluation mode."""
        self.model.eval()

    def cuda(self):
        """Transfers the parameters of the model to a CUDA device."""
        self.model.cuda()
