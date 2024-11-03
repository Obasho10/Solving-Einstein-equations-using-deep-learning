import torch
from train import train_pinn_subdomains
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



num_subdomains = 1
epochs = 1000
batch_size = 1
learning_rate = 0.0001
r0=10
rf=200

# Call the training function
models = train_pinn_subdomains(num_subdomains, r0,rf, epochs, batch_size, learning_rate,device)

for idx, model in enumerate(models):
    r_min = r0+(idx/num_subdomains)*(rf-r0)
    r_max = r0+((idx+1)/num_subdomains)*(rf-r0)
    file_name = f"pinn_model_subdomain_{r_min}_{r_max}.pt"
    torch.save(model.state_dict(), file_name)
    print(f"Model for subdomain {r_min}-{r_max} saved as {file_name}")
