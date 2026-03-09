import torch
from torch.utils.data import Dataset
import torchvision.models as models
import torchvision.transforms as transforms
import pandas as pd
import os
import random
from PIL import Image
from transformers import AutoTokenizer, AutoModel


class TwitterSentimentDataset(Dataset):
    def __init__(self, csv_path, max_samples=None,
                 cache_path="cache/bert_embeddings.pt",
                 image_folder="data/twitter/images"):

        # -----------------------------
        # Load and shuffle CSV
        # -----------------------------
        self.data = pd.read_csv(csv_path, header=None, encoding="latin-1")
        self.data = self.data.sample(frac=1.0, random_state=42).reset_index(drop=True)

        if max_samples is not None:
            self.data = self.data.iloc[:max_samples]

        self.texts = self.data[5].tolist()
        self.labels = self.data[0].apply(lambda x: 0 if x == 0 else 1).tolist()

        # -----------------------------
        # Dimensions
        # -----------------------------
        self.text_dim = 768
        self.audio_dim = 128
        self.visual_dim = 512

        # -----------------------------
        # BERT Caching
        # -----------------------------
        self.cache_path = cache_path
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        try:
            if os.path.exists(self.cache_path):
                print("Loading cached BERT embeddings...")
                self.text_embeddings = torch.load(self.cache_path)
            else:
                raise FileNotFoundError
        except Exception:
            print("Generating BERT embeddings (one-time)...")
            self.text_embeddings = self._generate_and_cache_embeddings()

        # -----------------------------
        # Load ResNet for Image Features
        # -----------------------------
        print("Loading pretrained ResNet18 for image embeddings...")
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = torch.nn.Identity()  # Remove final classification layer
        self.resnet.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        self.image_folder = image_folder
        if os.path.exists(self.image_folder):
            self.image_files = os.listdir(self.image_folder)
        else:
            print("Warning: Image folder not found. Using random visual features.")
            self.image_files = []

    # -----------------------------
    # Generate and cache BERT embeddings
    # -----------------------------
    def _generate_and_cache_embeddings(self):
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModel.from_pretrained("distilbert-base-uncased")
        model.eval()

        embeddings = []

        for text in self.texts:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )

            with torch.no_grad():
                outputs = model(**inputs)

            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)
            embeddings.append(cls_embedding)

        embeddings = torch.stack(embeddings)
        torch.save(embeddings, self.cache_path)

        return embeddings

    # -----------------------------
    def __len__(self):
        return len(self.texts)

    # -----------------------------
    def __getitem__(self, idx):

        # Text feature (cached BERT)
        text_feat = self.text_embeddings[idx]

        # Simulated audio (still dummy)
        audio_feat = torch.randn(self.audio_dim)

        # Real image feature if images exist
        if self.image_files:
            image_name = random.choice(self.image_files)
            image_path = os.path.join(self.image_folder, image_name)

            image = Image.open(image_path).convert("RGB")
            image = self.transform(image).unsqueeze(0)

            with torch.no_grad():
                visual_feat = self.resnet(image).squeeze(0)
        else:
            visual_feat = torch.randn(self.visual_dim)

        label = int(self.labels[idx])

        return text_feat, audio_feat, visual_feat, label