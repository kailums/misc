import torch.nn as nn
import torch._dynamo
from torch._functorch.aot_autograd import aot_module_simplified


class MLP(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(32, 64)
    self.ln = nn.LayerNorm(64)

  def forward(self, x):
    x = self.fc1(x)
    x = self.ln(x)
    #x = torch.nn.functional.gelu(x)
    return x


if __name__ == '__main__':
    device = torch.device(0)
    model = MLP()
    model.to(device)

    batch_size = 8
    input = torch.randn(batch_size, 32, device=device)

    def my_backend(gm, inputs):
        def my_compiler(gm, inputs):
            gm.print_readable()
            return gm.forward

        return aot_module_simplified(
                gm,
                inputs,
                fw_compiler=my_compiler
        )

    #import pdb;pdb.set_trace()
    #compiled_model = torch.compile(backend=my_backend, dynamic=True)(model)
    compiled_model = torch.compile(dynamic=True)(model)

    res = compiled_model(input)

    print('res shape: ', res.shape, ' dev: ', res.device)

