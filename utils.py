from model import MetricPINN
import torch.optim as optim

def compute_christoffel_symbols(metric, coords, device):
    batch_size, dim, _ = metric.shape
    inv_metric = torch.inverse(metric)  # Shape: [batch_size, dim, dim]

    # Initialize the Christoffel tensor with batch dimension
    christoffel = torch.zeros((batch_size, dim, dim, dim))

    # Compute Christoffel symbols by differentiating each element of metric w.r.t coords
    for mu in range(dim):
        for nu in range(dim):
              a1,a2,a3,a4=[],[],[],[]
              for alpha in range(dim):
                # List to accumulate the Christoffel terms for each batch element
                term1_list, term2_list, term3_list = [], [], []

                for b in range(batch_size):
                    # Compute the gradient of each element independently for the current batch index
                    term1 = grad(metric[b, nu, alpha], coords, create_graph=True)[0][b, mu]
                    term2 = grad(metric[b, mu, alpha], coords, create_graph=True)[0][b, nu]
                    term3 = grad(metric[b, mu, nu], coords, create_graph=True)[0][b, alpha]

                    # Collect terms for each batch element
                    term1_list.append(term1)
                    term2_list.append(term2)
                    term3_list.append(term3)

                # Stack the terms along the batch dimension to create batch tensors
                term1_tensor = torch.stack(term1_list)
                term2_tensor = torch.stack(term2_list)
                term3_tensor = torch.stack(term3_list)

                # Calculate the Christoffel symbol for the current indices
                a1.append(term1_tensor)
                a2.append(term2_tensor)
                a3.append(term3_tensor)
              a1t = torch.stack(a1).T
              a2t = torch.stack(a2).T
              a3t = torch.stack(a3).T

              christoffel[:, alpha, mu, nu] = 0.5 * torch.sum(
                    inv_metric[:, alpha, :] * (a1t + a2t - a3t), dim=-1
                )

    return christoffel

import torch

def compute_ricci_tensor(metric, coords, device):
    christoffel = compute_christoffel_symbols(metric, coords, device)
    batch_size, dim, _, _ = christoffel.shape
    ricci_tensor = torch.zeros(batch_size, dim, dim)

    for mu in range(dim):
        for nu in range(dim):
            term1 = 0
            term2 = 0
            term3 = 0
            term4 = 0
            for alpha in range(dim):
                # Compute term1 and term2 for each alpha
                term1_alpha = torch.autograd.grad(
                    christoffel[:, nu, mu, alpha].sum(), coords, create_graph=True
                )[0][:, alpha]

                term2_alpha = torch.autograd.grad(
                    christoffel[:, alpha, mu, alpha].sum(), coords, create_graph=True
                )[0][:, nu]

                term1 += term1_alpha
                term2 += term2_alpha

                for sigma in range(dim):
                  term3+=christoffel[:, alpha, sigma, alpha].sum()*christoffel[:, nu, mu, sigma].sum()
                  term4+=christoffel[:, sigma, nu, alpha].sum()*christoffel[:, alpha, mu, sigma].sum()

            # Add terms to Ricci tensor
            ricci_tensor[:, mu, nu] = term1 - term2+term3+term4

    return ricci_tensor

def compute_ricci_scalar(ricci_tensor, metric,device):
    inv_metric = torch.inverse(metric)
    ricci_scalar = torch.sum(inv_metric * ricci_tensor)
    return ricci_scalar

def eiloss(metric, coords, T,device, kappa=1):
    ricci_tensor = compute_ricci_tensor(metric, coords,device)
    print(ricci_tensor[0])
    ricci_tensor=ricci_tensor*10000000
    ricci_scalar = compute_ricci_scalar(ricci_tensor, metric,device)
    print(ricci_scalar)
    print(metric)

    einstein_tensor = ricci_tensor - 0.5 * ricci_scalar * metric

    residuals = einstein_tensor - kappa * T
    print(f"residullts {residuals}", f"einstein tensor : {einstein_tensor}"  )
    loss = torch.sum(residuals ** 2)

    return loss

def initmodels(num_models,learning_rate,device):
    models = []
    optimizers = []
    for _ in range(num_models):
        model = MetricPINN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        models.append(model)
        optimizers.append(optimizer)
    return models, optimizers
