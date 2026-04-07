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
 