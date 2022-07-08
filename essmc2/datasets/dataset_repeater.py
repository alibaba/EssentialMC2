from .registry import DATASETS


@DATASETS.register_class()
class DatasetRepeater(object):
    def __init__(self, dataset, num_repeats):
        super(DatasetRepeater, self).__init__()
        self.dataset = DATASETS.build(dataset)
        assert num_repeats >= 1
        self.num_repeats = num_repeats

    def __len__(self):
        return len(self.dataset) * self.num_repeats

    def __getitem__(self, idx):
        return self.dataset[idx % len(self.dataset)]

    def __repr__(self):
        return f"DatasetRepeater[{self.dataset}] * {self.num_repeats}"
