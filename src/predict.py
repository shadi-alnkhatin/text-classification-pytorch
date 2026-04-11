import torch
import torch.nn.functional as F
from src.model import TextClassificationModel
from src.dataset import text_pipeline, LABEL_NAMES, NUM_CLASSES

def load_model(path, vocab_size, embed_dim=128, dropout=0.3, device=None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TextClassificationModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_class=NUM_CLASSES,
        dropout=dropout
    )

    # Load the saved weights into the architecture
    # map_location ensures weights load correctly even if they were
    # saved on GPU but we're now running on CPU (or vice versa)
    model.load_state_dict(
        torch.load(path, map_location=device)
    )

    # Switch to evaluation mode:
    #   - Dropout is turned OFF (all values pass through)
    #   - Model is ready for inference
    model.eval()
    model.to(device)

    print(f"Model loaded from : {path}")
    print(f"Running on        : {device}")

    return model


# ─────────────────────────────────────────────
# 2.  PREDICT A SINGLE ARTICLE
# ─────────────────────────────────────────────

def predict(text, model, device=None):
  
    if device is None:
        device = next(model.parameters()).device

    # Convert raw text → list of token indices
    # e.g. "Oil prices surge" → [42, 891, 204]
    token_ids = text_pipeline(text)

    # Wrap in tensors
    # text tensor   : 1D tensor of all token ids
    # offsets tensor: [0] because there's only one article,
    #                 so it starts at position 0
    text_tensor   = torch.tensor(token_ids,  dtype=torch.int64).to(device)
    offset_tensor = torch.tensor([0],        dtype=torch.int64).to(device)

    # Run through the model without computing gradients
    # (we're not training, so we don't need them)
    with torch.no_grad():
        output = model(text_tensor, offset_tensor)  # shape: [1, 4]

    # Pick the index with the highest score
    predicted_index = output.argmax(dim=1).item()   # e.g. 1

    # Convert index → human-readable label
    return LABEL_NAMES[predicted_index]             # e.g. "Sports"


def predict_with_confidence(text, model, device=None):
 
    if device is None:
        device = next(model.parameters()).device

    token_ids     = text_pipeline(text)
    text_tensor   = torch.tensor(token_ids, dtype=torch.int64).to(device)
    offset_tensor = torch.tensor([0],       dtype=torch.int64).to(device)

    with torch.no_grad():
        output = model(text_tensor, offset_tensor)   # shape: [1, 4]

        # softmax turns raw scores into probabilities
        # dim=1 means we apply softmax across the 4 class scores
        probabilities = F.softmax(output, dim=1).squeeze()  # shape: [4]

    predicted_index = probabilities.argmax().item()

    confidence = {
        LABEL_NAMES[i]: round(probabilities[i].item(), 4)
        for i in range(NUM_CLASSES)
    }

    return {
        "prediction" : LABEL_NAMES[predicted_index],
        "confidence" : confidence
    }

def predict_batch(texts, model, device=None):
    """
    Classifies a list of articles efficiently in one forward pass.

    More efficient than calling predict() in a loop because all articles
    are processed together as one batch.

    Args:
        texts  : list of str — list of articles to classify
        model  : loaded TextClassificationModel
        device : "cpu" or "cuda"

    Returns:
        list of str — predicted label for each article
    """
    if device is None:
        device = next(model.parameters()).device

    # Build the flat tensor + offsets — same logic as collate_batch
    token_lists = [text_pipeline(t) for t in texts]

    # Compute starting position of each article
    offsets = [0]
    for tokens in token_lists[:-1]:
        offsets.append(offsets[-1] + len(tokens))

    # Flatten all token lists into one tensor
    flat_tokens   = [idx for tokens in token_lists for idx in tokens]
    text_tensor   = torch.tensor(flat_tokens, dtype=torch.int64).to(device)
    offset_tensor = torch.tensor(offsets,     dtype=torch.int64).to(device)

    with torch.no_grad():
        output = model(text_tensor, offset_tensor)   # shape: [len(texts), 4]

    predicted_indices = output.argmax(dim=1).tolist()

    return [LABEL_NAMES[i] for i in predicted_indices]



def show_predictions(texts, model, device=None):
    print("\n" + "─" * 72)
    print(f" {'#':>2}  {'Article (truncated)':<42}  {'Prediction':<20} {'Conf':>6}")
    print("─" * 72)

    for i, text in enumerate(texts, start=1):
        result     = predict_with_confidence(text, model, device)
        prediction = result["prediction"]
        confidence = result["confidence"][prediction]
        truncated  = (text[:40] + "...") if len(text) > 40 else text

        print(f" {i:>2}  {truncated:<42}  {prediction:<20} {confidence*100:>5.1f}%")

    print("─" * 72 + "\n")


if __name__ == "__main__":
    import os

    MODEL_PATH = "outputs/model.pth"
    VOCAB_SIZE = 95811

    if not os.path.exists(MODEL_PATH):
        print(f"No trained model found at {MODEL_PATH}")
        print("Run `python train.py` first to train and save the model.")
    else:
        model = load_model(MODEL_PATH, vocab_size=VOCAB_SIZE)

        sample_articles = [
            "Oil prices surge after OPEC cuts output",
            "Lakers defeat Celtics in overtime thriller",
            "New breakthrough in quantum computing announced",
            "Global markets react to US interest rate hike",
            "NASA signed a new sponsorship deal with a football club after discovering water on Mars, while stock markets reacted to the announcement",
            "Apple reported record profits this quarter while global markets reacted cautiously to inflation concerns and interest rate changes"
        ]

        show_predictions(sample_articles, model)