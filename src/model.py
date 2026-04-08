import torch
import torch.nn as nn


class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim=128, num_class=4, dropout=0.3):
        super().__init__()

        self.embedding = nn.EmbeddingBag(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            mode="mean",
            sparse=True
        )

        self.dropout = nn.Dropout(p=dropout)

        self.fc1 = nn.Linear(embed_dim, embed_dim // 2)

        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(embed_dim // 2, num_class)

        self._init_weights()


    def _init_weights(self):
        init_range = 0.5
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc1.weight.data.uniform_(-init_range, init_range)
        self.fc1.bias.data.zero_()
        self.fc2.weight.data.uniform_(-init_range, init_range)
        self.fc2.bias.data.zero_()


    def forward(self, text, offsets):

        # Step 1: Token indices → dense embedding vector per article
        embedded = self.embedding(text, offsets)

        # Step 2: Randomly zero out some values (only active during training)
        x = self.dropout(embedded)

        x = self.fc1(x)

        x = self.relu(x)

        x = self.dropout(x)

        # Step 3: Final projection to class scores
        x = self.fc2(x)

        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

