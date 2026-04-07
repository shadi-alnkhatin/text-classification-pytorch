import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import AG_NEWS
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import to_map_style_dataset
 
LABEL_NAMES = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Science/Technology"
}
 
NUM_CLASSES = len(LABEL_NAMES)

tokenizer = get_tokenizer("basic_english")

 
def _yield_tokens(data_iter):

    for _, text in data_iter:
        yield tokenizer(text)
 
 
def build_vocabulary():
    train_iter = AG_NEWS(split="train")
 
    vocab = build_vocab_from_iterator(_yield_tokens(train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    return vocab
 
 
vocab = build_vocabulary()
 

def text_pipeline(text):

    return vocab(tokenizer(text))
 
# Converts a raw AG_NEWS label (1-4) to a zero-indexed label (0-3).
def label_pipeline(label):
    return int(label) - 1
 
  
def collate_batch(batch):
    """
    Custom collate function for the DataLoader.
 
    Problem it solves:
        Articles have different lengths. We can't stack them into a
        fixed-size tensor directly. Instead we:
          1. Flatten all token lists into ONE long 1D tensor
          2. Track where each article starts using "offsets" because we need it for the embedding bag layer
    """
    label_list, text_list, offsets = [], [], [0]
 
    for label, text in batch:
        label_list.append(label_pipeline(label))
        processed_text = torch.tensor(text_pipeline(text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
 
    text_tensor = torch.cat(text_list)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
 
    label_tensor = torch.tensor(label_list, dtype=torch.int64)
 
    return label_tensor, text_tensor, offsets
 
 
def get_dataloaders(batch_size=64, val_ratio=0.05):
    """
    Loads AG_NEWS and returns three DataLoaders:
        train_loader, val_loader, test_loader
        
    Returns:
        train_loader : DataLoader
        val_loader   : DataLoader
        test_loader  : DataLoader
        vocab_size   : int  (needed to build the model)
    """
    train_iter, test_iter = AG_NEWS()
    train_dataset = to_map_style_dataset(train_iter)
    test_dataset  = to_map_style_dataset(test_iter)
 
    num_train = len(train_dataset)
    num_val   = int(num_train * val_ratio)
    num_train = num_train - num_val
 
    train_dataset, val_dataset = random_split(
        train_dataset,
        [num_train, num_val]
    )
 
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch
    )
 
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch
    )
 
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch
    )
 
    vocab_size = len(vocab)
 
    print(f"Vocabulary size : {vocab_size:,}")
    print(f"Training samples: {num_train:,}")
    print(f"Val samples     : {num_val:,}")
    print(f"Test samples    : {len(test_dataset):,}")
 
    return train_loader, val_loader, test_loader, vocab_size