import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from twitter_dataset import TwitterSentimentDataset


# -----------------------------
# Multimodal Fusion Model
# -----------------------------
class DemoMTVAF(nn.Module):
    def __init__(self, use_multimodal=True):
        super().__init__()
        self.use_multimodal = use_multimodal

        self.text_fc = nn.Linear(768, 128)

        if use_multimodal:
            self.audio_fc = nn.Linear(128, 64)
            self.visual_fc = nn.Linear(512, 128)
            self.classifier = nn.Linear(128 + 64 + 128, 2)
        else:
            self.classifier = nn.Linear(128, 2)

    def forward(self, text, audio, visual):
        t = self.text_fc(text)

        if self.use_multimodal:
            a = self.audio_fc(audio)
            v = self.visual_fc(visual)
            fused = torch.cat([t, a, v], dim=1)
        else:
            fused = t

        return self.classifier(fused)


# -----------------------------
# Training + Evaluation Function
# -----------------------------
def train_and_evaluate(model, train_loader, val_loader, model_name):

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 10
    patience = 2
    best_val_acc = 0
    trigger = 0

    epoch_losses = []

    print(f"\nStarting training for {model_name}\n")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for text, audio, visual, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(text, audio, visual)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_losses.append(total_loss)
        print(f"Epoch {epoch + 1} | Training Loss: {total_loss:.4f}")

        # ---------------- Validation ----------------
        model.eval()
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for text, audio, visual, labels in val_loader:
                outputs = model(text, audio, visual)
                preds = torch.argmax(outputs, dim=1)

                val_preds.extend(preds.tolist())
                val_labels.extend(labels.tolist())

        val_accuracy = accuracy_score(val_labels, val_preds)
        print(f"Validation Accuracy: {val_accuracy:.4f}\n")

        # ---------------- Early Stopping ----------------
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            trigger = 0
        else:
            trigger += 1
            if trigger >= patience:
                print("Early stopping triggered.\n")
                break

    # ---------------- Final Metrics ----------------
    precision, recall, f1, _ = precision_recall_fscore_support(
        val_labels, val_preds, average='macro'
    )

    print(f"\nFinal Validation Metrics ({model_name})")
    print(f"Accuracy : {best_val_acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    # ---------------- Confusion Matrix ----------------
    cm = confusion_matrix(val_labels, val_preds, labels=[0, 1])

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Negative", "Positive"]
    )

    disp.plot()
    plt.title(f"Confusion Matrix – {model_name}")
    plt.show()

    # ---------------- Error Analysis ----------------
    print("\nSample Misclassified Examples:\n")
    misclassified = [i for i in range(len(val_labels)) if val_labels[i] != val_preds[i]]

    for idx in misclassified[:5]:
        print(f"True Value: {val_labels[idx]} | Predicted Value: {val_preds[idx]}")

    # ---------------- Loss Plot ----------------
    plt.figure()
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title(f"Training Loss vs Epoch – {model_name}")
    plt.show()


# -----------------------------
# Main Function
# -----------------------------
def main():

    dataset = TwitterSentimentDataset(
        csv_path="data/twitter/training.1600000.processed.noemoticon.csv",
        max_samples=2000
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

    # ---------------- Text-Only Model ----------------
    text_only_model = DemoMTVAF(use_multimodal=False)
    train_and_evaluate(text_only_model, train_loader, val_loader, "Text-Only Model")

    # ---------------- Multimodal Model ----------------
    multimodal_model = DemoMTVAF(use_multimodal=True)
    train_and_evaluate(multimodal_model, train_loader, val_loader, "Multimodal Model")


if __name__ == "__main__":
    main()