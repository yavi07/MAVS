import torch
from torch.utils.data import Dataset

class DummyMultimodalDataset(Dataset):
    def __init__(self, num_samples=100):
        self.num_samples = num_samples

        self.text_dim = 768    
        self.audio_dim = 128    
        self.visual_dim = 512   
        self.num_classes = 3    

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        text = torch.randn(self.text_dim)
        audio = torch.randn(self.audio_dim)
        visual = torch.randn(self.visual_dim)
        label = torch.randint(0, self.num_classes, (1,)).item()

        return text, audio, visual, label
