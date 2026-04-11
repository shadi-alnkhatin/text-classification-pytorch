import torch
from src.dataset import get_dataloaders, LABEL_NAMES
from src.model   import TextClassificationModel
from src.train   import train_model, plot_history, evaluate
import torch.nn as nn


BATCH_SIZE  = 64
EMBED_DIM   = 128
NUM_EPOCHS  = 10
DROPOUT     = 0.3
SAVE_PATH   = "outputs/model.pth"

def main():

    print("\n[ 1 / 4 ]  Loading dataset...")
    train_loader, val_loader, test_loader, vocab_size = get_dataloaders(
        batch_size=BATCH_SIZE
    )

    print("\n[ 2 / 4 ]  Building model...")
    model = TextClassificationModel(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        num_class=len(LABEL_NAMES),
        dropout=DROPOUT
    )
    print(model)
    print(f"Total parameters: {model.count_parameters():,}")

    print("\n[ 3 / 4 ]  Training...")
    history = train_model(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        num_epochs   = NUM_EPOCHS,
        save_path    = SAVE_PATH
    )

    plot_history(history)
    print("\n[ 4 / 4 ]  Evaluating on test set...")

    model.load_state_dict(torch.load(SAVE_PATH))

    criterion = nn.CrossEntropyLoss()
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print(f"\n  Test Loss     : {test_loss:.4f}")
    print(f"  Test Accuracy : {test_acc:.4f}  ({test_acc*100:.2f}%)")
    print("\nDone! You can now run predictions with:")
    print("  python src/predict.py")


if __name__ == "__main__":
    main()