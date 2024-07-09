# Copyright (c) OpenMMLab. All rights reserved.
from .builder import build_dataset  # noqa: F401, F403
from .dota import DOTADataset  # noqa: F401, F403
from .dotav2 import DOTAV2Dataset
from .fair import FAIR1MDataset
from .hrsc import HRSCDataset  # noqa: F401, F403
from .tianzhi_car import TianzhiCarDataset
from .pipelines import *  # noqa: F401, F403

__all__ = ['DOTADataset', 'FAIR1MDataset', 'build_dataset', 'HRSCDataset', 'DOTAV2Dataset',
           'TianzhiCarDataset']
