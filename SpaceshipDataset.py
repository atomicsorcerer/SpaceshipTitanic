import numpy as np
import torch
from torch.utils.data import Dataset

import polars as pl


class SpaceshipDataset(Dataset):
	def __init__(self, file_path, transform=None, target_transform=None):
		self.labels = pl.read_csv(file_path).get_column("Transported").apply(lambda s: [1, 0] if s else [0, 1])

		features_tmp = pl.read_csv(file_path).drop("Transported").drop("Name")
		features_tmp = features_tmp.drop("PassengerId").with_columns(
			features_tmp.get_column("PassengerId").apply(lambda s: int(s.split("_")[0])).alias("GroupId"),
			features_tmp.get_column("PassengerId").apply(lambda s: int(s.split("_")[1])).alias("IntraGroupId"),
			features_tmp.get_column("Destination").apply(
				lambda s: 1 if s == "PSO J318.5-22" else (2 if "55 Cancri e" else (3 if "TRAPPIST-1e" else 0))
			),
			features_tmp.get_column("HomePlanet").apply(
				lambda s: 1 if s == "Europa" else (2 if "Mars" else (3 if "Earth" else 0))
			),
			features_tmp.get_column("Cabin").apply(
				lambda s: "ABCDEFGT".index(s.split("/")[0]) + 1
			).alias("RoomDeck"),
			features_tmp.get_column("Cabin").apply(lambda s: int(s.split("/")[1])).alias("RoomNumber"),
			features_tmp.get_column("Cabin").apply(lambda s: 0 if s.split("/")[2] == "P" else 1).alias("RoomSide"),
		).drop("Cabin").fill_null(strategy="zero")

		self.features = features_tmp

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):
		# Label: [TRUE, FALSE]: [1, 0]
		label = torch.tensor([self.labels[idx]], dtype=torch.float32)
		feature = torch.from_numpy(np.float32(self.features[idx].to_numpy()))

		return feature, label
