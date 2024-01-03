from torch import nn


class NeuralNetwork(nn.Module):
	def __init__(self):
		super().__init__()
		self.stack = nn.Sequential(
			nn.Linear(29, 2048),
			nn.Sigmoid(),
			nn.Linear(2048, 2048),
			nn.Sigmoid(),
			nn.Linear(2048, 1024),
			nn.Sigmoid(),
			nn.Linear(1024, 512),
			nn.Sigmoid(),
			nn.Linear(512, 256),
			nn.Sigmoid(),
			nn.Linear(256, 256),
			nn.Sigmoid(),
			nn.Linear(256, 2),
			nn.Softmax(dim=2)
		)

	def forward(self, x):
		# if len(x[0][0]) < 29:
		# 	print(list(x[0][0]))
		logits = self.stack(x)
		return logits
