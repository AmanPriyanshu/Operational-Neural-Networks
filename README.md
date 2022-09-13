# Operational-Neural-Networks
Operational Neural Networks (ONNs), which can be heterogeneous and encapsulate neurons with any set of operators to boost diversity and to learn highly complex and multi-modal functions or spaces with minimal network complexity and training data.

## Components:

Traditional neural networks employ `activation(W.x + b)`. Which can be further decomposed as `prod = W*x` --> `z = sum(prod, axis=0)` --> `a = activation(z)`. In ONNs these are referred to as components, basically nodal, pool, and activation operator.

### Nodal Operator:
Composed of: `multiplication, exponential, harmonic (sinusoid), quadratic function, Gaussian, Derivative of Gaussian (DoG), Laplacian of Gaussian (LoG), Hermitian, etc.`

### Pool Operator:
Composed of: `summation, n-correlation, maximum, median, etc.`

### Activation Operator:
Composed of standard neural network activations.

## Training:

Considering layerwise input to be the output of the previous layer, we can consider the concept of backpropagation to be sufficient to train such a neural network. However, this leads to the conundrum that the operator set of each neuron must be assigned before the application of backpropagation training. Making it a typical “Chicken and Egg” problem!

As to figure out which operators are the best, we'd need to experiment with all of them and since any change in pool/nodal operator would drastically reflect on every successive layer, it is necessary for it to be solved. The greedy iterative search (GIS) offers a solution, whereby, we attempt to find the best possible operator layerwise, instead of the entire network.