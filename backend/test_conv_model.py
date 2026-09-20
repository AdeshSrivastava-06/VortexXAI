import torch
import torch.nn as nn
import os

class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=self.padding
        )

    def forward(self, x, state=None):
        batch_size, _, height, width = x.size()
        if state is None:
            h = torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device, dtype=x.dtype)
            c = torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device, dtype=x.dtype)
        else:
            h, c = state

        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, g, o = torch.split(gates, self.hidden_channels, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, (h_next, c_next)

class ConvLSTMModel(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        # If in_channels == 4, we use a 1x1 conv or adapter if cell1 expects 2 channels
        self.channel_adapter = nn.Conv2d(in_channels, 2, kernel_size=1) if in_channels != 2 else None
        self.cell1 = ConvLSTMCell(in_channels=2, hidden_channels=32, kernel_size=3)
        self.cell2 = ConvLSTMCell(in_channels=32, hidden_channels=16, kernel_size=3)
        self.head = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, x):
        # x shape: (B, T, C, H, W) or (B, C, H, W)
        if x.dim() == 4:
            x = x.unsqueeze(1) # (B, 1, C, H, W)
        
        batch_size, seq_len, num_channels, height, width = x.size()
        
        state1 = None
        state2 = None
        
        for t in range(seq_len):
            x_t = x[:, t]
            if self.channel_adapter is not None:
                x_t = self.channel_adapter(x_t)
            h1, state1 = self.cell1(x_t, state1)
            h2, state2 = self.cell2(h1, state2)
            
        out = self.head(h2) # (B, 1, H, W)
        prob = torch.sigmoid(out.mean(dim=(-2, -1))).squeeze(-1) # (B, 1) or (B,)
        return prob

pth_path = os.path.join(os.path.dirname(__file__), "convlstm_bust_model.pth")
state_dict = torch.load(pth_path, map_location='cpu', weights_only=False)

model = ConvLSTMModel(in_channels=4)
# Load cell1, cell2, head from state_dict
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)

# Test forward pass with 4-channel input (B=1, T=3, C=4, H=13, W=13)
dummy_x = torch.randn(1, 3, 4, 13, 13)
with torch.no_grad():
    output = model(dummy_x)
print("Forward pass output:", output.item())
