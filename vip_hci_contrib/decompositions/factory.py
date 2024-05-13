"""
Author: Lukas Welzel
Date: 19.10.2023

Copyright (C) 2023 Lukas Welzel
"""

__author__ = "L. Welzel"
# __all__ = []  # __all__ is defined dynamically at the end of this file to include the dynamically created subclasses


import inspect

from vip_hci_contrib.decompositions.decomposition_codi_base import AbstractBaseTensorDecompositionCODI
from vip_hci_contrib.decompositions.polar_decomposition_codi_base import AbstractBasePolarTensorDecompositionCODI
from vip_hci_contrib.decompositions.annular_decomposition_codi_base import (
    AbstractBaseAnnularTensorDecompositionCODI, AbstractBaseSingleAnnulusTensorDecompositionCODI)
import vip_hci_contrib.decompositions.decomposition_bases as decomposition_bases


def get_classes_from_module(module):
    return {name: cls for name, cls in inspect.getmembers(module, inspect.isclass) if cls.__module__ == module.__name__}


def create_subclass(base_classes, **kwargs):
    """
    Dynamically create a subclass with the given name and base classes.
    The first base class in the list should be the one whose _psf_model method you want to use.
    """

    name = ""
    for base in base_classes[::-1]:
        name += base.__name__.replace("AbstractBase", "").replace("TensorDecomposition", "").replace("CODI", "")

    # Creating the class dynamically
    return type(name, tuple(base_classes), kwargs)


def get_dynamic_subclass_by_name(class_name):
    """
    Retrieves a dynamically created subclass by its name.

    Parameters:
    - class_name (str): The name of the dynamically created subclass.

    Returns:
    - The dynamically created subclass if found, or None if no class with the given name exists.
    """
    return globals().get(class_name, None)

decomposition_abstract_base_classes = get_classes_from_module(decomposition_bases)

base_classes = [
    AbstractBaseTensorDecompositionCODI,
    AbstractBasePolarTensorDecompositionCODI,
    AbstractBaseAnnularTensorDecompositionCODI,
    AbstractBaseSingleAnnulusTensorDecompositionCODI,
]

dynamic_subclasses = {}
for base_class in base_classes:
    for decomposition_base_classe in decomposition_abstract_base_classes.values():
        subclass = create_subclass([decomposition_base_classe, base_class])
        dynamic_subclasses[subclass.__name__] = subclass

globals().update(dynamic_subclasses)
# Dynamically update __all__ to include only the dynamically created classes
__all__ = list(dynamic_subclasses.keys())


if __name__ == "__main__":
    classes_in_globals = {name: obj for name, obj in globals().items() if isinstance(obj, type)}
    classes_in_globals_names = list(classes_in_globals.keys())

    print(classes_in_globals_names)
