import torch
from torch.utils.data import DataLoader

from NeuralNetwork import NeuralNetwork
from SpaceshipDataset import SpaceshipDataset
from utils import train, test

training_data = SpaceshipDataset("Data/training.csv")
test_data = SpaceshipDataset("Data/testing.csv")

# print(training_data.features.columns)
# print(training_data.__getitem__(1138))
# exit()

batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


model = NeuralNetwork()

cross_entropy_loss = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

epochs = 15
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, cross_entropy_loss, optimizer)
    test(test_dataloader, model, cross_entropy_loss)

print("Finished Training!\n")
do_save = input("Would you like to save the model? y/N -> ")
if do_save == "y":
    torch.save(model, "model.pth")
    print("Model Saved")
