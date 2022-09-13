import torch
from ONNlayers import ONNLinearLayer
from demo import MNISTDataset
from tqdm import tqdm

class GIS:
	def __init__(self, input_dim, neuron_vector, output_dim, training_dataloader, testing_dataloader):
		self.input_dim = input_dim
		self.neuron_vector = neuron_vector
		self.output_dim = output_dim
		self.training_dataloader = training_dataloader
		self.testing_dataloader = testing_dataloader

	def train_single_model(self, vec_to_analyze, previous_model, output_vec, nodal_param, pool_param, verbose=False, epochs=5):
		array_of_layers = []
		for layer_param in previous_model:
			array_of_layers.append(ONNLinearLayer(layer_param["input_dim"], layer_param["output_dim"], nodal_param=layer_param["nodal_param"], pool_param=layer_param["pool_param"]))
			array_of_layers.append(torch.nn.ReLU())
		array_of_layers.append(ONNLinearLayer(vec_to_analyze[0], vec_to_analyze[1], nodal_param=nodal_param, pool_param=pool_param))
		array_of_layers.append(torch.nn.ReLU())
		array_of_layers.append(torch.nn.Linear(output_vec[0], output_vec[1]))
		model = torch.nn.Sequential(*array_of_layers)
		model.train()
		optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
		criterion = torch.nn.CrossEntropyLoss()
		running_loss = 0.0
		running_acc = 0.0
		for epoch in range(epochs):
			for batch_idx, (batch_x, batch_y) in enumerate(self.training_dataloader):
				model.zero_grad()
				out = model(batch_x)
				loss = criterion(out, batch_y)
				optimizer.step()
				pred = torch.argmax(out, 1)
				acc = torch.mean((pred==batch_y).float()).item()
				running_acc += acc
				running_loss += loss.item()
				if verbose:
					print(nodal_param, pool_param, running_loss/(batch_idx+1), running_acc/(batch_idx+1))
		training_loss, training_acc = running_loss/(batch_idx+1), running_acc/(batch_idx+1)
		
		model.eval()
		running_loss = 0.0
		running_acc = 0.0
		for batch_idx, (batch_x, batch_y) in enumerate(self.testing_dataloader):
			out = model(batch_x)
			loss = criterion(out, batch_y)
			pred = torch.argmax(out, 1)
			acc = torch.mean((pred==batch_y).float()).item()
			running_acc += acc
			running_loss += loss.item()
		testing_loss, testing_acc = running_loss/(batch_idx+1), running_acc/(batch_idx+1)
		return training_loss, training_acc, testing_loss, testing_acc

	def param_iterator(self):
		for nodal_param in ["mul", "cubic", "harmonic", "exp", "sinc", "chirp"]:
			for pool_param in ["summation", "median", "max", "min"]:
				yield nodal_param, pool_param


	def run_single_layer(self, vec_to_analyze, previous_model, output_vec):
		best_params = {"input_dim":vec_to_analyze[0], "output_dim":vec_to_analyze[1], "nodal_param": "mul", "pool_param": "summation", "value": 1e6}
		search_space = [param for param in self.param_iterator()]
		for nodal_param, pool_param in tqdm(search_space, desc="Solving for layer- In:"+str(vec_to_analyze[0])+" Out:"+str(vec_to_analyze[1])):
			out = self.train_single_model(vec_to_analyze, previous_model, output_vec, nodal_param, pool_param)
			if out[-2]<best_params["value"]:
				best_params["nodal_param"]=nodal_param
				best_params["pool_param"]=pool_param
				best_params["value"]=out[-2]
		return best_params

	def iterate_through_the_layers(self):
		previous = []
		vec = [self.input_dim]
		layerwise_prop = []
		for neuron in self.neuron_vector:
			vec.append(neuron)
			best_props = self.run_single_layer(vec, layerwise_prop, [neuron, self.output_dim])
			layerwise_prop.append(best_props)
			vec = vec[1:]
		return layerwise_prop

	def generate_final_model(self, layerwise_prop):
		array_of_layers
		for layer_param in layerwise_prop:
			array_of_layers.append(ONNLinearLayer(layer_param["input_dim"], layer_param["output_dim"], nodal_param=layer_param["nodal_param"], pool_param=layer_param["pool_param"]))
			array_of_layers.append(torch.nn.ReLU())
		array_of_layers.append(torch.nn.Linear(layer_param["output_dim"], self.output_dim))
		model = torch.nn.Sequential(*array_of_layers)
		return model


def main(batch_size=1):
	input_dim = 784
	neuron_vector = [256, 128, 32]
	output_dim = 10
	training_dataset = MNISTDataset(p=0.01)
	testing_dataset = MNISTDataset("./data/MNIST_val.csv", p=0.01)
	training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size)
	testing_dataloader = torch.utils.data.DataLoader(testing_dataset, batch_size=batch_size)

	gis = GIS(input_dim, neuron_vector, output_dim, training_dataloader, testing_dataloader)
	layerwise_prop = gis.iterate_through_the_layers()
	model = gis.generate_final_model(layerwise_prop)
	print(model)

if __name__ == '__main__':
	main()