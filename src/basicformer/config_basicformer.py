r"""
This is the configuration file for BasicFormer, a 100% huggingface-compatible pytorch model written in pure, compact, elegant, easy-to-read style.
"""

from transformers import PretrainedConfig


class BasicFormerConfig(PretrainedConfig):
    r"""Configuration for BasicFormer.

    Parameters:
        - vocab_size: Total number of different tokens to embed.
        - embedding_size: Embedding size after input embedding.
        - hidden_size: The vector size for each token, usually equals to embedding_size.
        - num_hidden_layers: Number of hidden layers in the Transformer encoder.
        - num_attention_heads: Number of attention heads for each attention layer in the Transformer encoder.
        - intermediate_size: Dimension size of the "intermediate" (i.e., feed-forward) layer in the Transformer blocks.
        - max_position_embeddings: maximum number of positions in the input.
        - hidden_dropout_prob: Hidden dropout probability.
        - attention_dropout_prob: Attention dropout probability.
        - layer_norm_eps: Normalization epsilon for layer norm.
        - is_decoder: Flag that indicates a Decoder-only model. Conflicts with `is_encoder_decoder`.
        - is_encoder_decoder: Flag that indicates a Encoder-Decoder model. Conflicts with `is_decoder`.
        - bos_token_id: ID of start token.
        - eos_token_id: ID of end token.
        - pad_token_id: ID of padding token.
        - is_not_pad: A value for `attention_mask`, position of `attention_mask` with this value is not padded, thus will not be masked. It is usually `1`, unless tokenizer define it with different value.
        - initializer_range: Range of nn.Embedding weights in normal distribution.
    """

    model_type = "shift-former"

    def __init__(
        self,
        vocab_size: int = 32768,
        embedding_size: int = 768,
        hidden_size: int = 768,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 12,
        intermediate_size: int = 3052,
        max_position_embeddings: int = 1024,
        hidden_dropout_prob: float = 0.1,
        attention_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-6,
        is_decoder: bool = False,
        is_encoder_decoder: bool = False,
        bos_token_id: int = 0,
        eos_token_id: int = 2,
        pad_token_id: int = 1,
        is_not_pad: int = 1,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        assert not (is_decoder and is_encoder_decoder), (
            "Model cannot be both decoder-only and encoder-decoder."
        )
        self.is_decoder = is_decoder
        self.is_encoder_decoder = is_encoder_decoder
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.is_not_pad = is_not_pad
        self.initializer_range = initializer_range
