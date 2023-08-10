from torch import nn


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=16, embed_dim=768, in_channels=3):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=embed_dim,
                stride=patch_size,
                kernel_size=patch_size,
            ),
            nn.Flatten(start_dim=-2, end_dim=-1),
        )

    def forward(self, x):
        return self.embedding(x).permute(0, 2, 1)


class MultiheadSelfAttentionBlock(nn.Module):
    def __init__(self, embedding_dim=768, num_heads=12, attn_dropout=0.0):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape=embedding_dim)
        self.msa = nn.MultiheadAttention(
            num_heads=num_heads,
            embed_dim=embedding_dim,
            dropout=attn_dropout,
            batch_first=True,
        )

    def forward(self, x):
        x_ln = self.ln(x)
        x, _ = self.msa(query=x_ln, key=x_ln, value=x_ln, need_weights=False)
        return x


class MLPBlock(nn.Module):
    def __init__(self, embed_dim=768, dropout=0.1, mlp_size=3072):
        super().__init__()

        self.ln = nn.LayerNorm(normalized_shape=embed_dim)

        self.layers = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=mlp_size),
            nn.Dropout(p=dropout),
            nn.GELU(),
            nn.Linear(in_features=mlp_size, out_features=embed_dim),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        return self.layers(self.ln(x))


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        embedding_dim=768,
        num_heads=12,
        attn_dropout=0.0,
        dropout=0.1,
        mlp_size=3072,
    ):
        super().__init__()
        self.msa_block = MultiheadSelfAttentionBlock(
            embedding_dim, num_heads, attn_dropout
        )
        self.mlp_block = MLPBlock(embedding_dim, dropout, mlp_size)

    def forward(self, x):
        x = self.msa_block(x) + x
        x = self.mlp_block(x) + x
        return x


class ViT(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        in_channels: int = 3,
        patch_size: int = 16,
        num_transformer_layers: int = 12,
        embed_dim: int = 768,
        mlp_size: int = 3072,
        num_heads: int = 12,
        attn_dropout: float = 0.0,
        mlp_dropout: float = 0.1,
        embed_dropout: float = 0.1,
        num_classes: int = 1000,
    ):
        super().__init__()

        self.num_patches = (img_size * img_size) // (patch_size**2)

        self.class_token = nn.Parameter(
            data=torch.randn(1, 1, embed_dim), requires_grad=True
        )

        self.position_embedding = nn.Parameter(
            data=torch.randn(1, self.num_patches + 1, embed_dim), requires_grad=True
        )

        self.embed_dropout = nn.Dropout(p=embed_dropout)

        self.patch_embedding = PatchEmbedding(
            in_channels=in_channels, patch_size=patch_size, embed_dim=embed_dim
        )

        self.transformer_encoder = nn.Sequential(
            *[
                TransformerEncoderBlock(
                    embedding_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_size=mlp_size,
                    dropout=mlp_dropout,
                )
                for _ in range(num_transformer_layers)
            ]
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embed_dim),
            nn.Linear(in_features=embed_dim, out_features=num_classes),
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        class_token = self.class_token.expand(x.shape[0], -1, -1)

        x = torch.cat([class_token, x], dim=1)
        x = self.position_embedding + x

        x = self.embed_dropout(x)

        x = self.transformer_encoder(x)

        x = self.classifier(x[:, 0])

        return x
