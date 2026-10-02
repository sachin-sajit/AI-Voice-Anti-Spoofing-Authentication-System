import torch
import torch.nn as nn
import torch.nn.functional as F

class AntiSpoofCNN(nn.Module):
    """
    Lightweight 2D CNN Model for Voice Replay Attack Detection.
    Compatible across CPU, MPS (Apple Silicon), and CUDA devices.
    
    Inputs: Log-Mel Spectrogram tensor of shape (Batch, 1, N_MELS, Time_Steps)
    Outputs: Replay Probability tensor of shape (Batch, 1) in range [0, 1]
             0.0 = BONAFIDE / LIVE VOICE
             1.0 = SPOOF / REPLAY ATTACK
    """
    def __init__(self, n_mels: int = 64):
        super(AntiSpoofCNN, self).__init__()
        
        # Convolutional Block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolutional Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolutional Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolutional Block 4
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Classification Head (Global Average Pooling)
        self.dropout1 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(256, 128)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Conv Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Conv Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        
        # Global Average Pooling across spatial dimensions (H, W) -> shape: (Batch, 256)
        x = torch.mean(x, dim=(2, 3))
        
        # Fully Connected Layers
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        logits = self.fc2(x)
        
        return torch.sigmoid(logits)

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    model = AntiSpoofCNN()
    dummy_input = torch.randn(4, 1, 64, 94)
    out = model(dummy_input)
    print(f"Model created successfully. Output shape: {out.shape}, Total parameters: {count_parameters(model)}")
