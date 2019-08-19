from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .sample.ddd import DddDataset
from .sample.exdet import EXDetDataset
from .sample.ctdet import CTDetDataset
from .sample.evodet import EvoDetDataset
from .sample.heatdet import HeatDetDataset
from .sample.edgedet import EdgeDetDataset
from .sample.diagonaldet import DiagonalDetDataset
from .sample.multi_pose import MultiPoseDataset
from .sample.saccadedet import SaccadeDetDataset

from .dataset.coco import COCO
from .dataset.pascal import PascalVOC
from .dataset.kitti import KITTI
from .dataset.coco_hp import COCOHP


dataset_factory = {
  'coco': COCO,
  'pascal': PascalVOC,
  'kitti': KITTI,
  'coco_hp': COCOHP
}

_sample_factory = {
  'exdet': EXDetDataset,
  'ctdet': CTDetDataset,
  'evodet': EvoDetDataset,
  'edgedet': EdgeDetDataset,
  'ddd': DddDataset,
  'heatdet': HeatDetDataset,
  'multi_pose': MultiPoseDataset,
  'diagonaldet': DiagonalDetDataset,
  'saccadedet': SaccadeDetDataset
}


def get_dataset(dataset, task):
  class Dataset(dataset_factory[dataset], _sample_factory[task]):
    pass
  return Dataset
  
