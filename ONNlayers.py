import torch
import math

class ONNLinearLayer(torch.nn.Module):
	def __init__(self, size_in, size_out, nodal_param="mul", pool_param="summation"):
		super().__init__()
		self.size_in, self.size_out = size_in, size_out
		weights = torch.Tensor(size_out, size_in)
		self.weights = torch.nn.Parameter(weights)
		bias = torch.Tensor(size_out)
		self.bias = torch.nn.Parameter(bias)
		self.nodal_param = nodal_param # mul, cubic, harmonic, exp, sinc, chirp
		self.pool_param = pool_param   # summation, median, max, min

		torch.nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
		fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weights)
		bound = 1 / math.sqrt(fan_in)
		torch.nn.init.uniform_(self.bias, -bound, bound)

	def forward(self, x):
		if self.nodal_param=="mul":
			w_times_x = torch.mul(x, self.weights).t()
		elif self.nodal_param=="cubic":
			w_times_x = torch.mul(torch.pow(x, 3), self.weights).t()
		elif self.nodal_param=="harmonic":
			w_times_x = torch.sin(torch.mul(x, self.weights).t())
		elif self.nodal_param=="exp":
			w_times_x = torch.exp(torch.mul(x, self.weights)).t()-1
		elif self.nodal_param=="sinc":
			w_times_x = torch.sin(torch.mul(x, self.weights))/x
			w_times_x = w_times_x.t()
		elif self.nodal_param=="chirp":
			w_times_x = torch.sin(torch.mul(torch.pow(x, 2), self.weights)).t()
		else:
			raise Exception("Nodal Parameter must be one of the following: {mul, cubic, harmonic, exp, sinc, chirp}")
		if self.pool_param=="summation":
			out = torch.sum(w_times_x, axis=0, keepdim=True)
		elif self.pool_param=="median":
			out_ = torch.median(w_times_x, axis=0, keepdim=True)
			out = out_.values
		elif self.pool_param=="max":
			out_ = torch.median(w_times_x, axis=0, keepdim=True)
			out = out_.values
		elif self.pool_param=="min":
			out_ = torch.median(w_times_x, axis=0, keepdim=True)
			out = out_.values
		else:
			raise Exception("Pool Parameter must be one of the following: {summation, median, max, min}")
		return out

	def extra_repr(self):
		return 'input_features={}, output_features={}, bias={}, nodal_param={}, pool_param={}'.format(self.size_in, self.size_out, self.bias is not None, self.nodal_param, self.pool_param)

if __name__ == '__main__':
	for nodal_param in ["mul", "cubic", "harmonic", "exp", "sinc", "chirp"]:
		for pool_param in ["summation", "median", "max", "min"]:
			linear = MyLinearLayer(10, 20, nodal_param="chirp", pool_param="median")
			try:
				print(linear.weights.grad.shape)
			except:
				print("Grad not created")
			x = torch.randn(1, 10)
			y = linear(x)
			print(y.shape, nodal_param, pool_param)
			torch.sum(y).backward()
			print(linear.weights.grad.shape)