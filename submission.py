import numpy as np
import torch
from torch.utils.data import DataLoader

import polars as pl

from SpaceshipDataset import SubmissionDataset

submission_data = SubmissionDataset("Data/test.csv")

submission_dataloader = DataLoader(submission_data, shuffle=False)

model = torch.load('model.pth')

predictions_set = []

for X in submission_dataloader:
	pred = model(X)
	pred = pred.detach().numpy()[0][0]
	pred = np.where(pred == max(pred))[0][0]
	pred = "True" if pred == 0 else "False"

	predictions_set.append(pred)

template_submission = pl.read_csv("Submission/template_submission.csv")

final_submission = template_submission.drop("Transported").with_columns(
	pl.Series("Transported", predictions_set)
)
final_submission.write_csv("Submission/submission.csv")
