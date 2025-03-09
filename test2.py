import functorch as ft
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMLP(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=5, output_dim=10):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # Flatten MNIST images (28x28) into vectors of size 784.
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = nn.Linear(5, 5)
input = torch.randn(16, 5)
tangents = tuple([torch.rand_like(p) for p in model.parameters()])

tangents = [torch.zeros_like(p) for p in model.parameters()]
tangents[0][0] = 1
tangents = tuple(tangents)

print("parameteeers: ", list(model.parameters()))
print("tangeeents", tangents)

def loss_fn(p):
    # p is a tuple of parameters.
    out = func(params, buffers, input)
    loss = torch.sum(out ** 2)
    return loss

# Given a ``torch.nn.Module``, ``ft.make_functional_with_buffers`` extracts the state
# (``params`` and buffers) and returns a functional version of the model that
# can be invoked like a function.
# That is, the returned ``func`` can be invoked like
# ``func(params, buffers, input)``.
# ``ft.make_functional_with_buffers`` is analogous to the ``nn.Modules`` stateless API
# that you saw previously and we're working on consolidating the two.
func, params, buffers = ft.make_functional_with_buffers(model)

# Because ``jvp`` requires every input to be associated with a tangent, we need to
# create a new function that, when given the parameters, produces the output
def func_params_only(params):
    return func(params, buffers, input)

model_output, jvp_out = ft.jvp(func_params_only, (params,), (tangents,))
print(input)
print("eeeh")
print(model_output, jvp_out)