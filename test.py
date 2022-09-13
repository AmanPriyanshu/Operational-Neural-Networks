import torch

a = torch.randn(3,4, requires_grad=True)
	b = a.median(1)
print("a =", a)
print("b = ",b)

b.values.sum().backward()
print(a.grad)

a = torch.randn(3)
b = torch.randn(4, 3, requires_grad=True)
print(a.shape, b.shape)
print(torch.mul(a, b).shape)
print(a, "\n")
print(b, "\n")
y = torch.mul(a, b)
print(y.shape, y)
torch.sum(y).backward()
print(b.grad)