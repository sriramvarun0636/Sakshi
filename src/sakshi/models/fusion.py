"""The WP1 classifier: a fine-tuned text tower over frozen, cached CLIP tokens."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from sakshi.data import labels

BERTWEET_REVISION = "b349c1243407b0dcffeabb2337497477286e27ab"

HUMANITARIAN_CONDITIONAL = [name for name in labels.HUMANITARIAN if name != "not_humanitarian"]
HEAD_CLASSES = {
    "informative": labels.INFORMATIVE,
    "humanitarian": HUMANITARIAN_CONDITIONAL,
    "severity": labels.SEVERITY,
    "disaster_type": labels.DISASTER_TYPE,
    "informative_text": labels.INFORMATIVE,
    "humanitarian_text": labels.HUMANITARIAN,
    "informative_image": labels.INFORMATIVE,
    "humanitarian_image": labels.HUMANITARIAN,
}


@dataclass(frozen=True)
class FusionConfig:
    text_encoder: str = "vinai/bertweet-base"
    text_revision: str = BERTWEET_REVISION
    text_hidden: int = 768
    image_hidden: int = 768
    image_tokens: int = 50
    d_model: int = 512
    cross_attention_layers: int = 2
    attention_heads: int = 8
    ffn_dim: int = 2048
    dropout: float = 0.1
    head_hidden: int = 256
    head_dropout: float = 0.2
    gate: bool = True
    auxiliary_heads: bool = True

    def __post_init__(self) -> None:
        if self.d_model < 1 or self.d_model % self.attention_heads:
            raise ValueError("d_model must be positive and divisible by attention_heads")
        if self.cross_attention_layers < 0 or self.image_tokens < 1:
            raise ValueError("cross_attention_layers must be nonnegative and image_tokens positive")
        if not 0 <= self.dropout < 1 or not 0 <= self.head_dropout < 1:
            raise ValueError("dropout probabilities must lie in [0, 1)")


class CrossAttentionBlock(nn.Module):
    """Update both streams from the same block inputs, then apply separate feed-forwards."""

    def __init__(self, config: FusionConfig) -> None:
        super().__init__()
        dim = config.d_model
        self.text_norm = nn.LayerNorm(dim)
        self.image_norm = nn.LayerNorm(dim)
        self.text_to_image = nn.MultiheadAttention(dim, config.attention_heads, dropout=config.dropout, batch_first=True)
        self.image_to_text = nn.MultiheadAttention(dim, config.attention_heads, dropout=config.dropout, batch_first=True)
        self.text_ff_norm = nn.LayerNorm(dim)
        self.image_ff_norm = nn.LayerNorm(dim)
        self.text_ff = nn.Sequential(
            nn.Linear(dim, config.ffn_dim), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(config.ffn_dim, dim)
        )
        self.image_ff = nn.Sequential(
            nn.Linear(dim, config.ffn_dim), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(config.ffn_dim, dim)
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, text: Tensor, image: Tensor, text_mask: Tensor, image_mask: Tensor) -> tuple[Tensor, Tensor]:
        text_in, image_in = text, image
        text_norm, image_norm = self.text_norm(text_in), self.image_norm(image_in)
        text_attn, _ = self.text_to_image(text_norm, image_norm, image_norm, key_padding_mask=~image_mask, need_weights=False)
        image_attn, _ = self.image_to_text(image_norm, text_norm, text_norm, key_padding_mask=~text_mask, need_weights=False)
        text = text_in + self.dropout(text_attn)
        image = image_in + self.dropout(image_attn)
        return text + self.dropout(self.text_ff(self.text_ff_norm(text))), image + self.dropout(
            self.image_ff(self.image_ff_norm(image))
        )


def _head(dim: int, hidden: int, classes: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, classes))


class FusionClassifier(nn.Module):
    """Accept padded text IDs and cached 50-token image features with explicit presence masks.

    The text encoder is called only for rows with usable text. An injected encoder allows
    small offline tests; production uses :meth:`from_pretrained`.
    """

    def __init__(self, text_encoder: nn.Module, config: FusionConfig | None = None) -> None:
        super().__init__()
        self.config = config or FusionConfig()
        self.text_encoder = text_encoder
        hidden = getattr(getattr(text_encoder, "config", None), "hidden_size", None)
        if hidden is not None and hidden != self.config.text_hidden:
            raise ValueError(f"text encoder has hidden size {hidden}, expected {self.config.text_hidden}")
        dim = self.config.d_model
        self.text_projection = nn.Sequential(nn.Linear(self.config.text_hidden, dim), nn.LayerNorm(dim))
        self.image_projection = nn.Sequential(nn.Linear(self.config.image_hidden, dim), nn.LayerNorm(dim))
        self.text_embedding = nn.Parameter(torch.zeros(dim))
        self.image_embedding = nn.Parameter(torch.zeros(dim))
        self.image_positions = nn.Parameter(torch.empty(self.config.image_tokens, dim))
        nn.init.normal_(self.image_positions, std=0.02)
        self.no_text = nn.Parameter(torch.empty(dim))
        self.no_image = nn.Parameter(torch.empty(dim))
        nn.init.normal_(self.no_text, std=0.02)
        nn.init.normal_(self.no_image, std=0.02)
        self.blocks = nn.ModuleList(CrossAttentionBlock(self.config) for _ in range(self.config.cross_attention_layers))
        self.text_output_norm = nn.LayerNorm(dim)
        self.image_output_norm = nn.LayerNorm(dim)
        self.gate_network = nn.Sequential(nn.Linear(3 * dim, 256), nn.GELU(), nn.Linear(256, 2))
        selected = (
            HEAD_CLASSES
            if self.config.auxiliary_heads
            else {k: v for k, v in HEAD_CLASSES.items() if "_" not in k or k == "disaster_type"}
        )
        self.heads = nn.ModuleDict(
            {
                name: _head(dim, self.config.head_hidden, len(classes), self.config.head_dropout)
                for name, classes in selected.items()
            }
        )

    @classmethod
    def from_pretrained(cls, config: FusionConfig | None = None) -> FusionClassifier:
        from transformers import AutoModel

        config = config or FusionConfig()
        return cls(AutoModel.from_pretrained(config.text_encoder, revision=config.text_revision), config)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        image_tokens: Tensor,
        has_text: Tensor,
        has_image: Tensor,
    ) -> dict[str, Tensor]:
        config = self.config
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attention_mask must have matching B×L shapes")
        batch, length = input_ids.shape
        if length < 1 or image_tokens.shape != (batch, config.image_tokens, config.image_hidden):
            raise ValueError("image_tokens must have shape B×image_tokens×image_hidden, and text length must be positive")
        if has_text.shape != (batch,) or has_image.shape != (batch,):
            raise ValueError("modality masks must have shape B")
        has_text, has_image = has_text.bool(), has_image.bool()
        if torch.any(~(has_text | has_image)):
            raise ValueError("each row needs text or image")
        text_mask = attention_mask.bool() & has_text[:, None]
        if torch.any(has_text & ~text_mask.any(dim=1)):
            raise ValueError("rows with text need at least one unmasked token")
        text_hidden = torch.zeros(
            batch, length, config.text_hidden, device=input_ids.device, dtype=self.text_projection[0].weight.dtype
        )
        if torch.any(has_text):
            indices = torch.where(has_text)[0]
            encoded = self.text_encoder(input_ids=input_ids[indices], attention_mask=attention_mask[indices]).last_hidden_state
            text_hidden = text_hidden.index_copy(0, indices, encoded.to(dtype=text_hidden.dtype))
        text = self.text_projection(text_hidden) + self.text_embedding
        image = self.image_projection(image_tokens.to(dtype=self.image_projection[0].weight.dtype))
        image = image + self.image_embedding + self.image_positions

        # A missing stream has exactly one valid, learned token in its padded sequence.
        text = torch.where(has_text[:, None, None], text, self.no_text.view(1, 1, -1))
        image = torch.where(has_image[:, None, None], image, self.no_image.view(1, 1, -1))
        text_mask[:, 0] = True
        image_mask = has_image[:, None].expand(-1, config.image_tokens).clone()
        image_mask[:, 0] = True
        for block in self.blocks:
            text, image = block(text, image, text_mask, image_mask)
        text_pool = self.text_output_norm(text)[:, 0]
        image_pool = self.image_output_norm(image)[:, 0]

        if config.gate:
            scores = self.gate_network(torch.cat((text_pool, image_pool, text_pool * image_pool), dim=-1))
        else:
            scores = torch.zeros(batch, 2, dtype=text_pool.dtype, device=text_pool.device)
        scores = scores.masked_fill(~torch.stack((has_text, has_image), dim=1), torch.finfo(scores.dtype).min)
        alpha = scores.softmax(dim=-1)
        fused = alpha[:, :1] * text_pool + alpha[:, 1:] * image_pool
        results = {"alpha": alpha}
        for name, head in self.heads.items():
            source = (
                text_pool
                if name.endswith("_text")
                else image_pool
                if name.endswith("_image") or name in ("severity", "disaster_type")
                else fused
            )
            results[name] = head(source)
        return results
