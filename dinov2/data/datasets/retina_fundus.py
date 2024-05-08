import csv
from enum import Enum
import logging
import os
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from .extended import ExtendedVisionDataset

logger = logging.getLogger("dinov2")
_Target = int

class _Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"  # NOTE: torchvision does not support the test split

    @property
    def length(self) -> int:
        split_lengths = {
            _Split.TRAIN: 1_281_167,
            _Split.VAL: 50_000,
            _Split.TEST: 100_000,
        }
        return split_lengths[self]

    def get_dirname(self, class_id: Optional[str] = None) -> str:
        return self.value if class_id is None else os.path.join(self.value, class_id)

    def get_image_relpath(self, actual_index: int, class_id: Optional[str] = None) -> str:
        dirname = self.get_dirname(class_id)
        if self == _Split.TRAIN:
            basename = f"{class_id}_{actual_index}"
        else:  # self in (_Split.VAL, _Split.TEST):
            basename = f"ILSVRC2012_{self.value}_{actual_index:08d}"
        return os.path.join(dirname, basename + ".JPEG")

    def parse_image_relpath(self, image_relpath: str) -> Tuple[str, int]:
        assert self != _Split.TEST
        dirname, filename = os.path.split(image_relpath)
        class_id = os.path.split(dirname)[-1]
        basename, _ = os.path.splitext(filename)
        actual_index = int(basename.split("_")[-1])
        return class_id, actual_index

class Fundus(ExtendedVisionDataset):
    Target = Union[_Target]
    Split = Union[_Split]

    def __init__(
        self,
        *,
        split: "Fundus.Split",
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._split = split

        self._entries = None
        self._class_ids = None
        self._class_names = None

    @property
    def split(self) -> "Fundus.Split":
        return self._split

    def __len__(self) -> int:
        print(self._split)
        return self._split.length

    def find_class_id(self, class_index: int) -> str:
        class_ids = self._class_ids
        return str(class_ids[class_index])

    def find_class_name(self, class_index: int) -> str:
        class_names = self._class_names
        return str(class_names[class_index])

    def get_image_data(self, index: int) -> bytes:
        actual_index = index + 1  # or however you determine the actual index
        class_id = self.find_class_id(index)

        image_relpath = self.split.get_image_relpath(actual_index, class_id)
        image_full_path = os.path.join(self.root, image_relpath)
        with open(image_full_path, mode="rb") as f:
            image_data = f.read()
        return image_data

    def get_target(self, index: int) -> Optional[Target]:
        if self.split == _Split.TEST:
            return None
        class_index = index  # or however you determine the class index
        return int(class_index)

    def get_class_id(self, index: int) -> Optional[str]:
        if self.split == _Split.TEST:
            return None
        return str(index)  # or however you determine the class ID

    def get_class_name(self, index: int) -> Optional[str]:
        if self.split == _Split.TEST:
            return None
        return "Class Name"  # or however you determine the class name
