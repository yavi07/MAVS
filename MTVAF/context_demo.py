import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from context_dataset import ContextMultimodalDataset
from context_model import ContextFusionModel


dataset = ContextMultimodalDataset()
loader = DataLoader(dataset, batch_size=1)

model = ContextFusionModel()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("Running Context Multimodal Demo\n")

for epoch in range(3):

    total_loss = 0

    for post, caption, tags, image, label in loader:

        optimizer.zero_grad()

        output = model(
            post.squeeze(0),
            caption.squeeze(0),
            tags.squeeze(0),
            image.squeeze(0)
        )

        loss = criterion(output.unsqueeze(0), label)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print("Epoch", epoch+1, "Loss:", total_loss)