import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.profiler

# Example model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# Example dataset
data = torch.randn(1000, 10)
labels = torch.randn(1000, 1)
dataset = TensorDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model, loss function, and optimizer
model = SimpleModel().cuda()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Simplified Profiling
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
) as prof:
    for epoch in range(2):  # Run for 2 epochs
        for batch_data, batch_labels in dataloader:
            batch_data, batch_labels = batch_data.cuda(), batch_labels.cuda()
            optimizer.zero_grad()
            output = model(batch_data)
            loss = criterion(output, batch_labels)
            loss.backward()
            optimizer.step()
            prof.step()  # Step the profiler to record the current batch

# Print profiling results
print(prof.key_averages().table(
    sort_by="self_cuda_time_total", row_limit=-1))