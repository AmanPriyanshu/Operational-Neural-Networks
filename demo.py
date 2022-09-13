import torch
import pandas as pd

class MNISTDataset(torch.utils.data.Dataset):
	def __init__(self, path="./data/mnist_train.csv", p=1.0):
		self.path = path
		self.data = pd.read_csv(self.path)
		self.data = self.data.values
		self.data = self.data[:int(p*len(self.data))]
		self.labels = self.data.T[0]
		self.data = self.data.T[1:].T

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		x = torch.from_numpy(self.data[idx])
		return x, self.labels[idx]

if __name__ == '__main__':
	mnist_train_dataset = MNISTDataset()
	mnist_train_dataloader = torch.utils.data.DataLoader(mnist_train_dataset, batch_size=1)
	for batch_x, batch_y in mnist_train_dataloader:
		print(batch_x)
		print(batch_y)
		break