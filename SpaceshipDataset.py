import numpy as np
import torch
from torch.utils.data import Dataset
import polars as pl


class SpaceshipDataset(Dataset):
	def __init__(self, file_path, transform=None, target_transform=None):
		# Set labels based on if the passenger was transported
		self.labels = pl.read_csv(file_path).get_column("Transported").apply(lambda s: [1, 0] if s else [0, 1])

		# Import dataset and remove 'Transported' column (used for labels) and 'Name,' because it is not relevant
		features_tmp = pl.read_csv(file_path).drop("Transported").drop("Name")
		features_tmp = (features_tmp.drop("PassengerId").with_columns(
			features_tmp.get_column("PassengerId").apply(lambda s: int(s.split("_")[0])).alias("GroupId"),
			features_tmp.get_column("PassengerId").apply(lambda s: int(s.split("_")[1])).alias("IntraGroupId"),
			# One hot encode
			*features_tmp.get_column("Destination").to_dummies().get_columns(),
			*features_tmp.get_column("HomePlanet").to_dummies().get_columns(),
			*features_tmp.get_column("Cabin").apply(
				lambda s: "ABCDEFGT".index(s.split("/")[0])
			).alias("RoomDeck").to_dummies().get_columns(),
			features_tmp.get_column("Cabin").apply(lambda s: int(s.split("/")[1])).alias("RoomNumber"),
			# One hot encode
			*features_tmp.get_column("Cabin").apply(lambda s: s.split("/")[2]).alias("RoomSide").to_dummies().get_columns(),
			*features_tmp.get_column("CryoSleep").to_dummies().get_columns(),
			*features_tmp.get_column("VIP").to_dummies().get_columns(),
		).drop("Cabin").drop("Destination").drop("Destination_null").drop("HomePlanet").drop("HomePlanet_null")
						.drop("RoomDeck_null").drop("RoomSide_null").drop("CryoSleep").drop("CryoSleep_null")
						.drop("VIP").drop("VIP_null"))

		# Fill any remaining null values with 0
		features_tmp = features_tmp.fill_null(strategy="zero")

		self.features = features_tmp

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):
		# Label: [TRUE, FALSE]: [1, 0]
		label = torch.tensor([self.labels[idx]], dtype=torch.float32)
		feature = torch.from_numpy(np.float32(self.features[idx].to_numpy()))

		return feature, label


class SubmissionDataset(Dataset):
	def __init__(self, file_path, transform=None, target_transform=None):
		# Set labels based on if the passenger was transported
		self.labels = pl.read_csv(file_path)

		# Import dataset and remove 'Transported' column (used for labels) and 'Name,' because it is not relevant
		features_tmp = pl.read_csv(file_path).drop("Name")
		features_tmp = features_tmp.drop("PassengerId").with_columns(
			features_tmp.get_column("PassengerId").apply(lambda s: int(s.split("_")[0])).alias("GroupId"),
			features_tmp.get_column("PassengerId").apply(lambda s: int(s.split("_")[1])).alias("IntraGroupId"),
			# One hot encode
			*features_tmp.get_column("Destination").to_dummies().get_columns(),
			*features_tmp.get_column("HomePlanet").to_dummies().get_columns(),
			*features_tmp.get_column("Cabin").apply(
				lambda s: "ABCDEFGT".index(s.split("/")[0])
			).alias("RoomDeck").to_dummies().get_columns(),
			features_tmp.get_column("Cabin").apply(lambda s: int(s.split("/")[1])).alias("RoomNumber"),
			# One hot encode
			*features_tmp.get_column("Cabin").apply(lambda s: s.split("/")[2]).alias("RoomSide").to_dummies().get_columns(),
			*features_tmp.get_column("CryoSleep").to_dummies().get_columns(),
			*features_tmp.get_column("VIP").to_dummies().get_columns(),
		).drop("Cabin").drop("Destination").drop("Destination_null").drop("HomePlanet").drop("HomePlanet_null").drop("RoomDeck_null").drop("RoomSide_null").drop("CryoSleep").drop("CryoSleep_null").drop("VIP").drop("VIP_null")

		# Fill any remaining null values with 0
		features_tmp = features_tmp.fill_null(strategy="zero")

		self.features = features_tmp

	def __len__(self):
		return len(self.features.get_column("GroupId"))

	def __getitem__(self, idx):
		feature = torch.from_numpy(np.float32(self.features[idx].to_numpy()))

		return feature
