from mmrotate.datasets import DOTADataset
from .builder import ROTATED_DATASETS


@ROTATED_DATASETS.register_module()
class DOTAV2Dataset(DOTADataset):
    '''DOTA v2 dataset'''
    CLASSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
               'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank', 'soccer-ball-field',
               'roundabout', 'harbor', 'swimming-pool', 'helicopter',
               'container-crane', 'airport', 'helipad')