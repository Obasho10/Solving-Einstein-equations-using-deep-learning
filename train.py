import torch
import torch.optim as optim
import numpy as np
from utils import initmodels
from utils import eiloss
def train_pinn_subdomains(num_subdomains, r0,rf, epochs, batch_size,device, learning_rate=0.001, kappa=1):
    models, optimizers = initmodels(num_subdomains,learning_rate,device)

    for subdomain_idx in range(num_subdomains):
        r_min = r0+(subdomain_idx/num_subdomains)*(rf-r0)
        r_max = r0+((subdomain_idx+1)/num_subdomains)*(rf-r0)
        model = models[subdomain_idx]
        optimizer = optimizers[subdomain_idx]
        print(f"Training model for subdomain {subdomain_idx + 1} with r-range: [{r_min}, {r_max}]")
        for epoch in range(epochs):
            total_loss = 0.0
            r_samples = torch.FloatTensor(batch_size).uniform_(r_min, r_max)     # Sample r from subdomain range
            t_samples = torch.zeros(batch_size)                                  # Fixed time coordinate (optional)
            theta_samples = torch.FloatTensor(batch_size).uniform_(0, torch.pi)  # Sample θ from [0, π]
            phi_samples = torch.FloatTensor(batch_size).uniform_(0, 2 * torch.pi) # Sample φ from [0, 2π]
            coords = torch.stack([t_samples, r_samples, theta_samples, phi_samples], dim=1).requires_grad_(True)
            T = torch.zeros((batch_size, 4, 4)).to(device)  # Placeholder for the energy-momentum tensor
            metric = model(coords.to(device))
            loss = eiloss(metric, coords, T, kappa)
            print("loss")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            avg_loss = total_loss / batch_size
            print(f"Epoch [{epoch + 1}/{epochs}], Subdomain [{subdomain_idx + 1}], Loss: {avg_loss:.4f}")

    return models
