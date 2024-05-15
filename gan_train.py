import math
import os
import numpy as np
import matplotlib.pyplot as plt
from models import * 
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from sklearn.metrics import log_loss


def predict(gen,samples,epoch):
  with torch.no_grad():
    generated = gen(samples)
    generated = generated.cpu().detach()
    checkpoint = f"checkpoint_{epoch}"
    plt.imshow(generated[0][0].reshape(32, 32), cmap="gray")
    plt.savefig(f"checkpoints/{checkpoint}.png")


if __name__ == "__main__":

    device = "cuda"

    torch.manual_seed(42)
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    

    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    train_set = ImageFolder('data/', transform=transform)
    
    batch_size = 32 
    train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True
        )
    
    lr = 0.0001
    num_epochs = 10000
    loss_function = nn.HingeEmbeddingLoss()
    
    real_samples, labels = next(iter(train_loader))
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for n, (real_samples,labels) in enumerate(train_loader):
            real_samples = real_samples.to(device=device)
            real_samples_labels = torch.ones((real_samples.size(0),1)).to(device=device)
            latent_space_samples = torch.randn((batch_size,100)).to(device=device)
            generated_samples = generator(latent_space_samples)
            generated_samples_labels = torch.zeros((latent_space_samples.size(0),1)).to(device=device)
            all_samples = torch.cat((real_samples,generated_samples))
            all_samples_labels = torch.cat(
                (real_samples_labels,generated_samples_labels)
            )
            discriminator.zero_grad()
            output_discriminator = discriminator(all_samples)
            loss_discriminator = loss_function(
                output_discriminator, all_samples_labels
            )
            loss_discriminator.backward()
            optimizer_discriminator.step()
        
            latent_space_samples = torch.randn((real_samples.size(0), 100)).to(device=device)
        
            generator.zero_grad()
            generated_samples = generator(latent_space_samples)
            output_discriminator_generated = discriminator(generated_samples)
            loss_generator = loss_function(
                output_discriminator_generated, real_samples_labels
            )
            loss_generator.backward()
            optimizer_generator.step()
        
            if epoch %1==0 and n == batch_size -1:
              print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
              print(f"Epoch: {epoch} Loss G.: {loss_generator}")
            if epoch %50==0 and n == batch_size -1:
                predict(generator,latent_space_samples,epoch)
                discriminator.save_model(f"model_checkpoints/discriminator_{epoch}.pth")
                generator.save_model(f"model_checkpoints/generator_{epoch}.pth")

    discriminator.save_model(f"final/discriminator_{num_epochs}.pth")
    generator.save_model(f"/final/generator_{num_epochs}.pth")
