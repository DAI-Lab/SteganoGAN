# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import Mock, call, patch

# To test this module we need to create a folder with some images inside to be loaded as dataset


class TestDataLoader(TestCase):
    """
    METHOD:
        __init__(self)

    VALIDATE:
        * attributes
        * DEFAULT_TRANSFORM when we don't pass a transformation

    TODO:
        * test that DataLoader it's initiated with the params that we pass.
        * test that transform it's DEFAULT_TRANSFORM if we don't pass transform.
        * mock torch.utils.data.DataLoader
    """


class TestImageFolder(TestCase):
    """
    METHOD:
        __init__(self, path, transform, limit=np.inf)

    VALIDATE:
        * attributes
        * length

    TODO:
        * test that the limit it's not above the images loaded. (method __len__ should do that)
        * mock torchvision.datasets.ImageFolder
    """

