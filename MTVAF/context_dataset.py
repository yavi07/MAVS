import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.models as models
from transformers import AutoTokenizer, AutoModel
from PIL import Image


class ContextMultimodalDataset(Dataset):

    def __init__(self):

        # Example dataset (5 samples)
        self.data = [
            {
                "post": "I love this place!",
                "caption": "Amazing beach sunset",
                "tags": "#vacation #happy",
                "image": "data/images/img1.jpg",
                "label": 1
            },
            {
                "post": "Worst service ever",
                "caption": "Restaurant visit",
                "tags": "#angry #badservice",
                "image": "data/images/img2.jpg",
                "label": 0
            },
            {
                "post": "What a beautiful day",
                "caption": "Morning walk",
                "tags": "#sunshine #nature",
                "image": "data/images/img3.jpg",
                "label": 1
            },
            {
                "post": "Totally disappointed",
                "caption": "Concert cancelled",
                "tags": "#sad #wasteoftime",
                "image": "data/images/img4.jpg",
                "label": 0
            },
            {
                "post": "Best vacation ever",
                "caption": "Mountain view",
                "tags": "#travel #peaceful",
                "image": "data/images/img5.jpg",
                "label": 1
            }
        ]

        # Load BERT
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.bert = AutoModel.from_pretrained("distilbert-base-uncased")
        self.bert.eval()

        # Load ResNet
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = torch.nn.Identity()
        self.resnet.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def encode_text(self, text):

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=64
        )

        with torch.no_grad():
            outputs = self.bert(**inputs)

        return outputs.last_hidden_state[:, 0, :].squeeze(0)

    def encode_image(self, image_path):

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            features = self.resnet(image)

        return features.squeeze(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        sample = self.data[idx]

        post_feat = self.encode_text(sample["post"])
        caption_feat = self.encode_text(sample["caption"])
        tag_feat = self.encode_text(sample["tags"])

        image_feat = self.encode_image(sample["image"])

        label = torch.tensor(sample["label"])

        return post_feat, caption_feat, tag_feat, image_feat, label