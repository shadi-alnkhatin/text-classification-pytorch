import torch
import torch.nn as nn

class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim=128, num_class=4, dropout=0.3):
        super().__init__()

        self.embedding = nn.EmbeddingBag(
            vocab_size,
            embed_dim,
            mode="mean",
            sparse=False
        )

        self.dropout = nn.Dropout(p=dropout)

        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(embed_dim, num_class)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, text, offsets):

        x = self.embedding(text, offsets)
        x = self.dropout(x)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)

        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

