# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import Mock, call, patch

# To test this module we need to create a folder with some images inside to be loaded as dataset


class TestDataLoader(TestCase):
    """
    METHOD:
        __init__(self, ...)

    VALIDATE:
        * DEFAULT_TRANSFORM when we don't pass a transformation
        * Arguments passed to super().__init__

    MOCK:
        * torch.utils.data.DataLoader.__init__   (only the method!)
    """


class TestImageFolder(TestCase):
    """
    METHOD:
        __init__(self, path, transform, limit=np.inf)

    VALIDATE:
        * limit attribute
        * Arguments passed to super().__init__

    MOCK:
        * torchvision.datasets.ImageFolder.__init__   (only the method!)
    """

    """
    METHOD:
        __len__(self)

    VALIDATE:
        * lenght returned

    TEST CASES:
        * lenght_gt_limit
        * length_lt_limit
    """
