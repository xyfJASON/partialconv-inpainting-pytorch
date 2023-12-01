class AverageMeter:
    """
    Computes and stores the average and current value

    Adapted from https://github.com/huggingface/pytorch-image-models/blob/main/timm/utils/metrics.py
    """
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.__init__()

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class KeyValueAverageMeter:
    def __init__(self, keys):
        super().__init__()
        self.keys = keys
        self.val = {k: 0 for k in keys}
        self.avg = {k: 0 for k in keys}
        self.sum = {k: 0 for k in keys}
        self.count = 0

    def reset(self):
        self.__init__(self.keys)

    def update(self, kvs, n=1):
        assert set(kvs.keys()) == set(self.keys)
        self.count += n
        for k, val in kvs.items():
            self.val[k] = val
            self.sum[k] += val * n
            self.avg[k] = self.sum[k] / self.count
