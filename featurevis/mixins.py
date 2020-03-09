from __future__ import annotations
import os
import tempfile
from typing import Callable, Dict
from string import ascii_letters
from random import choice

import torch
from nnfabrik.utility.dj_helpers import make_hash

from . import integration


class TrainedEnsembleModelTemplateMixin:
    definition = """
    # contains ensemble ids
    -> self.dataset_table
    ensemble_hash                   : char(32)      # the hash of the ensemble
    ---
    ensemble_comment        = ''    : varchar(256)  # a short comment describing the ensemble
    """

    class Member:
        definition = """
        # contains assignments of trained models to a specific ensemble id
        -> master
        -> master.trained_model_table
        """

        insert: Callable[[Dict], None]

    dataset_table = None
    trained_model_table = None
    ensemble_model_class = integration.EnsembleModel

    insert1: Callable[[Dict], None]

    def create_ensemble(self, key, comment=""):
        if len(self.dataset_table() & key) != 1:
            raise ValueError("Provided key not sufficient to restrict dataset table to one entry!")
        dataset_key = (self.dataset_table().proj() & key).fetch1()
        models = (self.trained_model_table().proj() & key).fetch(as_dict=True)
        primary_key = dict(dataset_key, ensemble_hash=integration.hash_list_of_dictionaries(models))
        self.insert1(dict(primary_key, ensemble_comment=comment))
        self.Member().insert([{**primary_key, **m} for m in models])

    def load_model(self, key=None):
        return self._load_ensemble_model(key=key)

    def _load_ensemble_model(self, key=None):
        model_keys = (self.trained_model_table() & key).fetch(as_dict=True)
        dataloaders, models = tuple(
            list(x) for x in zip(*[self.trained_model_table().load_model(key=k) for k in model_keys])
        )
        return dataloaders[0], self.ensemble_model_class(*models)


class CSRFV1SelectorTemplateMixin:
    definition = """
    # contains information that can be used to map a neuron's id to its corresponding integer position in the output of
    # the model. 
    -> self.dataset_table
    neuron_id       : smallint unsigned # unique neuron identifier
    ---
    neuron_position : smallint unsigned # integer position of the neuron in the model's output 
    session_id      : varchar(13)       # unique session identifier
    """

    dataset_table = None
    dataset_fn = "csrf_v1"
    constrained_output_model = integration.ConstrainedOutputModel

    insert: Callable[[Dict], None]
    __and__: Callable[[Dict], CSRFV1SelectorTemplateMixin]

    @property
    def _key_source(self):
        return self.dataset_table() & dict(dataset_fn=self.dataset_fn)

    def make(self, key, get_mappings=integration.get_mappings):
        dataset_config = (self.dataset_table() & key).fetch1("dataset_config")
        mappings = get_mappings(dataset_config, key)
        self.insert(mappings)

    def get_output_selected_model(self, model, key):
        neuron_pos, session_id = (self & key).fetch1("neuron_position", "session_id")
        return self.constrained_output_model(model, neuron_pos, forward_kwargs=dict(data_key=session_id))


class MEIMethodMixin:
    definition = """
    # contains methods for generating MEIs and their configurations.
    method_fn                           : varchar(64)   # name of the method function
    method_hash                         : varchar(32)   # hash of the method config
    ---
    method_config                       : longblob      # method configuration object
    method_ts       = CURRENT_TIMESTAMP : timestamp     # UTZ timestamp at time of insertion
    """

    insert1: Callable[[Dict], None]
    __and__: Callable[[Dict], MEIMethodMixin]

    seed_table = None
    import_func = staticmethod(integration.import_module)

    def add_method(self, method_fn, method_config):
        self.insert1(dict(method_fn=method_fn, method_hash=make_hash(method_config), method_config=method_config))

    def generate_mei(self, dataloader, model, key, seed):
        method_fn, method_config = (self & key).fetch1("method_fn", "method_config")
        method_fn = self.import_func(method_fn)
        mei, score, output = method_fn(dataloader, model, method_config, seed)
        return dict(key, mei=mei, score=score, output=output)


class MEISeedMixin:
    definition = """
    # contains seeds used to make the MEI generation process reproducible
    mei_seed    : tinyint unsigned  # MEI seed
    """


class MEITemplateMixin:
    definition = """
    # contains maximally exciting images (MEIs)
    -> self.method_table
    -> self.trained_model_table
    -> self.selector_table
    -> self.seed_table
    ---
    mei                 : attach@minio  # the MEI as a tensor
    score               : float         # some score depending on the used method function
    output              : attach@minio  # object returned by the method function
    """

    trained_model_table = None
    selector_table = None
    method_table = None
    seed_table = None
    model_loader_class = integration.ModelLoader
    save_func = staticmethod(torch.save)
    temp_dir_func = tempfile.TemporaryDirectory

    insert1: Callable[[Dict], None]

    def __init__(self, *args, cache_size_limit=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_loader = self.model_loader_class(self.trained_model_table, cache_size_limit=cache_size_limit)

    def make(self, key):
        dataloaders, model = self.model_loader.load(key=key)
        seed = (self.seed_table() & key).fetch1("mei_seed")
        output_selected_model = self.selector_table().get_output_selected_model(model, key)
        mei_entity = self.method_table().generate_mei(dataloaders, output_selected_model, key, seed)
        self._insert_mei(mei_entity)

    def _insert_mei(self, mei_entity):
        """Saves the MEI to a temporary directory and inserts the prepared entity into the table."""
        with self.temp_dir_func() as temp_dir:
            for name in ("mei", "output"):
                self._save_to_disk(mei_entity, temp_dir, name)
            self.insert1(mei_entity)

    def _save_to_disk(self, mei_entity, temp_dir, name):
        data = mei_entity.pop(name)
        filename = name + "_" + self._create_random_filename() + ".pth.tar"
        filepath = os.path.join(temp_dir, filename)
        self.save_func(data, filepath)
        mei_entity[name] = filepath

    @staticmethod
    def _create_random_filename(length=32):
        return "".join(choice(ascii_letters) for _ in range(length))
