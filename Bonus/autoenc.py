import torch
import torchvision
from torch import nn
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import pickle

device = torch.device(torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu")
print(f"Running Autoencoder on: {device}")

fmnist_data = torchvision.datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)
loader = torch.utils.data.DataLoader(fmnist_data, batch_size=64, shuffle=True)


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Sigmoid()  # Keeps pixel values bounded between 0 and 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.view(-1, 1, 28, 28)


ae_model = Autoencoder().to(device)
optim = torch.optim.Adam(ae_model.parameters(), lr=0.001)
lossfn = nn.MSELoss()  #Autoencoders commonly use MSE according to what I read, hence not using Cross Entropy and softmax, hence we applied a sigmoid in the sequential itself, if you notice

epochs = 10  # Kept at 3 so you finish in under 2 minutes
train_loss_list = []

print("Training Autoencoder")
for epoch in range(epochs):
    ae_model.train()
    running_loss = 0.0

    for x, _ in loader:
        x = x.to(device)
        optim.zero_grad()
        outputs = ae_model(x)

        loss = lossfn(outputs, x)
        loss.backward()
        optim.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    train_loss_list.append(avg_loss)
    print(f"Autoencoder Epoch {epoch + 1} | Loss: {avg_loss:.4f}")

print("Generating Image and Weights...")
ae_model.eval()

images, _ = next(iter(loader))
images = images.to(device)
with torch.no_grad():
    reconstructed = ae_model(images)

images = images.cpu()
reconstructed = reconstructed.cpu()

fig, axes = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True, figsize=(10, 4))
plt.suptitle("Autoencoder: Original vs Reconstructed")

for i in range(5):
    axes[0, i].imshow(images[i].squeeze(), cmap='gray')
    axes[0, i].axis('off')
    if i == 0: axes[0, i].set_title("Original")

    axes[1, i].imshow(reconstructed[i].squeeze(), cmap='gray')
    axes[1, i].axis('off')
    if i == 0: axes[1, i].set_title("Reconstructed")

plt.tight_layout()
plt.savefig("autoencoder_results.png")
plt.show()

weights = ae_model.state_dict()
pickle.dump(weights, open("autoencoder_weights.pkl", "wb"))
print("autoencoder_results.png and autoencoder_weights.pkl saved successfully!")