import torch
from net import Net

model_to_load = "./alchemistic_directory/baseline/models/best.pth.tar"

net = Net()
net.load_state_dict(torch.load(model_to_load)["state_dict"])
net.eval()

jit_module = torch.jit.trace(net, torch.rand(1, 1, 28, 28))
torch.jit.save(jit_module, "jit_module.pth")
