import torch
import torch.utils.data as data
import numpy as np
class TimeSeriesDataset(data.Dataset):
  def __init__(self, dataset):
    super(TimeSeriesDataset, self).__init__()
    self.feature= torch.FloatTensor(dataset[:,1:]).unsqueeze_(1)
    self.label = torch.LongTensor(np.squeeze(dataset[:,0:1], axis=1))
    self.n_classes = len(set(self.label.tolist()))
    self.fea_dim   = self.feature.shape[-1]

  def __getitem__(self, idx):
    feature = self.feature[idx]
    label   = self.label[idx]
    return feature, label

  def toGPU(self,device):
    self.feature = self.feature.to(device)
    self.label = self.label.to(device)