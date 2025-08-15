import torch.nn as nn

class ForeverDataset(nn.Module):
    def __init__(self, dataset_loader, device, batchsize):
        super(ForeverDataset, self).__init__()
        self.dataset_loader = dataset_loader
        self.iter_dat = None
        self.batchsize = batchsize
        self.device = device
        self.num = len(self.dataset_loader)
    def init_data(self):
        self.iter_dat = iter(self.dataset_loader)

    def get_len(self):
        return self.num

    def get_data(self):
        if self.iter_dat is None:
            self.init_data()
        try:
            [[data_source, _, _], label_source, idx] = self.iter_dat.next()
        except StopIteration:
            self.init_data()
            [[data_source, _, _], label_source, idx] = self.iter_dat.next()
        if data_source.size(0) < self.batchsize:
            self.init_data()
            [[data_source, _, _], label_source, idx] = self.iter_dat.next()
        data_source, label_source = data_source.to(self.device), label_source.to(self.device)
        return  data_source, label_source, idx