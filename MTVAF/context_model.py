import torch
import torch.nn as nn


class ContextFusionModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.post_fc = nn.Linear(768, 128)
        self.caption_fc = nn.Linear(768, 128)
        self.tag_fc = nn.Linear(768, 128)

        self.image_fc = nn.Linear(512, 128)

        self.classifier = nn.Linear(512, 2)

    def forward(self, post, caption, tags, image):

        p = self.post_fc(post)
        c = self.caption_fc(caption)
        t = self.tag_fc(tags)
        i = self.image_fc(image)

        fused = torch.cat([p, c, t, i], dim=0)

        return self.classifier(fused)