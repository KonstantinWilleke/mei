"""This module contains the main tables and table templates used in the MEI generation process."""

import datajoint as dj

from nnfabrik.main import Dataset, schema
from . import mixins
from nnfabrik.utility.dj_helpers import make_hash
from . import integration
from mlutils.data.datasets import StaticImageSet


class TrainedEnsembleModelTemplate(mixins.TrainedEnsembleModelTemplateMixin, dj.Manual):
    """TrainedEnsembleModel table template.

    To create a functional "TrainedEnsembleModel" table, create a new class that inherits from this template and
    decorate it with your preferred Datajoint schema. Next assign the trained model table of your choosing to the class
    variable called "trained_model_table". By default the created table will point to the "Dataset" table in the
    Datajoint schema called "nnfabrik.main". This behaviour can be changed by overwriting the class attribute called
    "dataset_table".
    """

    dataset_table = Dataset

    class Member(mixins.TrainedEnsembleModelTemplateMixin.Member, dj.Part):
        """Member table template."""


class CSRFV1SelectorTemplate(mixins.CSRFV1SelectorTemplateMixin, dj.Computed):
        definition = """
        # contains assignments of trained models to a specific ensemble id
        -> master
        -> master.trained_model_table
        """

    def create_ensemble(self, key, model_keys=None):
        """Creates a new ensemble and inserts it into the table.

        Args:
            key: A dictionary representing a key that must be sufficient to restrict the dataset table to one entry. The
                models that are in the trained model table after restricting it with the provided key will be part of
                the ensemble.

        Returns:
            None.
        """
        print("creating ensemble")
        if len(self.dataset_table() & key) != 1:
            raise ValueError("Provided key not sufficient to restrict dataset table to one entry!")
        dataset_key = (self.dataset_table().proj() & key).fetch1()
        if model_keys is None:
            models = (self.trained_model_table().proj() & key).fetch(as_dict=True)
        else:
            print("model dictionties were passed - creating ensemble with {} models".format(len(model_keys)))
            models = model_keys
        ensemble_table_key = dict(dataset_key, ensemble_hash=integration.hash_list_of_dictionaries(models))
        self.insert1(ensemble_table_key)

        self.Member().insert([{**ensemble_table_key, **m} for m in models])

    def load_model(self, key=None):
        """Wrapper to preserve the interface of the trained model table."""
        return integration.load_ensemble_model(self.Member, self.trained_model_table, key=key)


class MouseSelectorTemplate(dj.Computed):
    """CSRF V1 selector table template.

    To create a functional "CSRFV1Selector" table, create a new class that inherits from this template and decorate it
    with your preferred Datajoint schema. By default, the created table will point to the "Dataset" table in the
    Datajoint schema called "nnfabrik.main". This behavior can be changed by overwriting the class attribute called
    "dataset_table".
    """

    dataset_table = Dataset

    definition = """
    # contains information that can be used to map a neuron's id to its corresponding integer position in the output of
    # the model. 
    -> self.dataset_table
    neuron_id       : smallint unsigned # unique neuron identifier
    ---
    neuron_position : smallint unsigned # integer position of the neuron in the model's output 
    session_id      : varchar(13)       # unique session identifier
    """

    _key_source = Dataset & dict(dataset_fn="mouse_static_loaders")

    def make(self, key):
        dataset_config = (Dataset & key).fetch1("dataset_config")

        path = dataset_config["paths"][0]
        dat = StaticImageSet(path, 'images', 'responses')
        neuron_ids = dat.neurons.unit_ids

        data_key = path.split('static')[-1].split('.')[0].replace('preproc', '')

        mappings = []
        for neuron_pos, neuron_id in enumerate(neuron_ids):
            mappings.append(dict(key, neuron_id=neuron_id, neuron_position=neuron_pos, session_id=data_key))

        self.insert(mappings)

    def get_output_selected_model(self, model, key):
        neuron_pos, session_id = (self & key).fetch1("neuron_position", "session_id")
        return integration.get_output_selected_model(neuron_pos, session_id, model)


@schema
class MEISeed(mixins.MEISeedMixin, dj.Lookup):
    """Seed table for MEI method."""


@schema
class MEIMethod(mixins.MEIMethodMixin, dj.Lookup):
    """Table that contains MEI methods and their configurations."""


class MEITemplate(mixins.MEITemplateMixin, dj.Computed):
    """MEI table template.

    To create a functional "MEI" table, create a new class that inherits from this template and decorate it with your
    preferred Datajoint schema. Next assign your trained model (or trained ensemble model) and your selector table to
    the class variables called "trained_model_table" and "selector_table". By default, the created table will point to
    the "MEIMethod" table in the Datajoint schema called "nnfabrik.main". This behavior can be changed by overwriting
    the class attribute called "method_table".
    """

    method_table = MEIMethod
    seed_table = MEISeed
