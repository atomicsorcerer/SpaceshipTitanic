from torch import nn


class NeuralNetwork(nn.Module):
	def __init__(self):
		super().__init__()
		self.stack = nn.Sequential(
			nn.Linear(15, 1024),
			nn.Sigmoid(),
			nn.Linear(1024, 1024),
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
		logits = self.stack(x)
		return logits
