import torch
from torch import optim

from minetorch.miner import Miner
from minetorch.metrics import MultiClassesClassificationMetricWithLogic
from torchvision import datasets, transforms

from net import Net


train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
        ),
    ),
    batch_size=128,
    shuffle=True,
)

val_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "./data",
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
        ),
    ),
    batch_size=128,
    shuffle=True,
)

model = Net()

trainer = Miner(
    alchemistic_directory="./alchemistic_directory",
    code="baseline",
    model=model,
    optimizer=optim.SGD(model.parameters(), lr=0.01),
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    loss_func=torch.nn.CrossEntropyLoss(),
    plugins=[MultiClassesClassificationMetricWithLogic()],
)

trainer.train()
