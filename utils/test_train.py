import torch
from torch.utils.data import DataLoader, TensorDataset

from models.model_loader import get_resnet50
from utils.train import train_model

# create tiny fake dataset
dummy_x = torch.randn(8, 3, 224, 224)
dummy_y = torch.randint(0, 6, (8,))

dataset = TensorDataset(dummy_x, dummy_y)
loader = DataLoader(dataset, batch_size=2)

# load model
model = get_resnet50(6)

# train for 1 epoch only
train_model(model, loader, loader, epochs=1)