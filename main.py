import torch
from torch.utils.data import DataLoader, Dataset

from SpaceshipDataset import SpaceshipDataset

training_data = SpaceshipDataset("train.csv")

print(training_data.__getitem__(0))
print(training_data.__getitem__(1))
print(training_data.__getitem__(2))
# batch_size = 64
#
# train_dataloader = DataLoader(training_data, batch_size=batch_size)
# test_dataloader = DataLoader(test_data, batch_size=batch_size)
#
