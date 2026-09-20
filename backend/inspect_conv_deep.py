import torch
import os

pth_path = os.path.join(os.path.dirname(__file__), "convlstm_bust_model.pth")
data = torch.load(pth_path, map_location='cpu', weights_only=False)

print("Type of loaded object:", type(data))
if isinstance(data, dict):
    print("Keys:", data.keys())
    for k, v in data.items():
        if torch.is_tensor(v):
            print(k, v.shape, v.dtype, "mean:", v.float().mean().item(), "std:", v.float().std().item())
