import os
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from src.model import TextClassificationModel

def train(model, dataloader, optimizer, criterion, device):
 
    model.train()

    total_loss    = 0.0
    total_correct = 0
    total_count   = 0

    for labels, text, offsets in dataloader:
        labels  = labels.to(device)
        text    = text.to(device)
        offsets = offsets.to(device)

        optimizer.zero_grad()

        output = model(text, offsets)

        loss = criterion(output, labels)

        loss.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

        optimizer.step()

        total_loss    += loss.item() * labels.size(0)
        total_correct += (output.argmax(dim=1) == labels).sum().item()
        total_count   += labels.size(0)

    avg_loss = total_loss / total_count
    accuracy = total_correct / total_count

    return avg_loss, accuracy




def evaluate(model, dataloader, criterion, device):

    model.eval()

    total_loss    = 0.0
    total_correct = 0
    total_count   = 0

    with torch.no_grad():
        for labels, text, offsets in dataloader:

            labels  = labels.to(device)
            text    = text.to(device)
            offsets = offsets.to(device)

            output = model(text, offsets)
            loss   = criterion(output, labels)

            total_loss    += loss.item() * labels.size(0)
            total_correct += (output.argmax(dim=1) == labels).sum().item()
            total_count   += labels.size(0)

    avg_loss = total_loss / total_count
    accuracy = total_correct / total_count

    return avg_loss, accuracy


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs   = 10,
    save_path    = "outputs/model.pth",
    device       = None):
 
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Training on: {device}")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    history = {
        "train_loss": [],
        "val_loss":   [],
        "train_acc":  [],
        "val_acc":    []
    }

    best_val_acc  = 0.0
    best_epoch    = 0

    # Make sure the output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print("=" * 65)
    print(f"  {'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | "
          f"{'Val Loss':>8} | {'Val Acc':>7} | {'Time':>6}")
    print("=" * 65)

    # ── Epoch loop ────────────────────────────────────────────────────
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        train_loss, train_acc = train(
            model, train_loader, optimizer, criterion, device
        )

        # Evaluate on validation set
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - start_time

        print(f"  {epoch:>5} | {train_loss:>10.4f} | {train_acc:>9.4f} | "
              f"{val_loss:>8.4f} | {val_acc:>7.4f} | {elapsed:>5.1f}s")

        # Only save when validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            torch.save(model.state_dict(), save_path)

    print("=" * 65)
    print(f"  Best model: epoch {best_epoch} | val accuracy: {best_val_acc:.4f}")
    print(f"  Saved to  : {save_path}")
    print("=" * 65)

    return history



def plot_history(history, save_path="outputs/plots/training_curves.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    ax1.plot(epochs, history["train_loss"], "b-o", label="Train loss", markersize=4)
    ax1.plot(epochs, history["val_loss"],   "r-o", label="Val loss",   markersize=4)
    ax1.set_title("Loss over epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(epochs, history["train_acc"], "b-o", label="Train acc", markersize=4)
    ax2.plot(epochs, history["val_acc"],   "r-o", label="Val acc",   markersize=4)
    ax2.set_title("Accuracy over epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Training History — AG_NEWS Text Classifier", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Training curves saved to: {save_path}")