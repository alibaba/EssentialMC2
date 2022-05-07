from .registry import DATASETS


@DATASETS.register_class()
class DatasetRepeater(object):
    def __init__(self, dataset, repeat_num):
        super(DatasetRepeater, self).__init__()
        self.dataset = DATASETS.build(dataset)
        assert repeat_num >= 2
        self.repeat_num = repeat_num

    def __len__(self):
        return len(self.dataset) * self.repeat_num

    def __getitem__(self, idx):
        return self.dataset[idx % len(self.dataset)]
