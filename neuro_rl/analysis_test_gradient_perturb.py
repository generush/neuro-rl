import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

a_out_temp = torch.ones((100, 12), requires_grad=True, device='cuda:0')
mu = a_out_temp * 2
sigma = mu + 1
mu.backward(torch.ones_like(mu))


model = nn.Sequential(  # a dummy model
    nn.Conv2d(1, 1, 3),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten()
)

sample_img = torch.rand(1, 5, 5)  # a dummy input
sample_label = 0

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)
device = "cpu"

sample_img = sample_img.to(device)
sample_img.requires_grad = True

prediction = model(sample_img.unsqueeze(dim=0))
cost = criterion(prediction, torch.tensor([sample_label]).to(device))
optimizer.zero_grad()
cost.backward()
print(sample_label)
print(sample_img.shape)

print(sample_img.grad.shape)
print(sample_img.grad)

plt.imshow(sample_img.detach().cpu().squeeze(), cmap='gray')
plt.show()



print(sample_img.grad)