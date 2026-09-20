import torch
import torch.nn as nn
import os

pth_path = os.path.join(os.path.dirname(__file__), "convlstm_bust_model.pth")
state_dict = torch.load(pth_path, map_location='cpu', weights_only=False)

for k, v in state_dict.items():
    print(f"{k}: shape={v.shape}, dtype={v.dtype}")

# Let's inspect cell1 and cell2:
# cell1.conv.weight: torch.Size([128, 34, 3, 3]) -> 128 = 4 * 32 gates. 34 = input_dim + hidden_dim? If hidden_dim=32, input_dim=2? Or is input_dim=4 and hidden_dim=30?
# cell2.conv.weight: torch.Size([64, 48, 3, 3]) -> 64 = 4 * 16 gates. 48 = 32 (cell1 hidden) + 16 (cell2 hidden).
# head.0.weight: torch.Size([16, 16, 3, 3]) -> Conv2d(16, 16, kernel_size=3, padding=1)
# head.2.weight: torch.Size([1, 16, 1, 1]) -> Conv2d(16, 1, kernel_size=1)
