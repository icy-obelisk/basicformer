r"""
This is the model file for BasicFormer, a 100% huggingface-compatible pytorch model written in pure, compact, elegant, easy-to-read style.
"""

import torch
from torch import nn
from torch.nn import functional as F
from transformers import (
    DynamicCache,
    EncoderDecoderCache,
    GenerationMixin,
    PreTrainedModel,
)
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithCrossAttentions,
    BaseModelOutputWithPast,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutput,
    CausalLMOutputWithPast,
    MaskedLMOutput,
    Seq2SeqLMOutput,
)

from .config_basicformer import BasicFormerConfig


class BasicFormerEmbedding(nn.Module):
    def __init__(self, config: BasicFormerConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_size)
        self.projection = (
            nn.Identity()
            if config.embedding_size == config.hidden_size
            else nn.Linear(config.embedding_size, config.hidden_size)
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Calculate token embeddings. (b * s) -> (b * s * e)
        input_embeds = self.embedding(input_ids)
        # Calculate projection. (b * s * e) -> (b * s * h)
        input_embeds = self.projection(input_embeds)
        # Combine token embeddings and positional encodings.
        input_embeds = self.norm(input_embeds)
        input_embeds = self.dropout(input_embeds)
        return input_embeds


class BasicFormerAttention(nn.Module):
    def __init__(
        self,
        config: BasicFormerConfig,
    ):
        super().__init__()
        self.config = config
        self.query = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.key = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.value = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.attention_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        # Calculate size of attention head
        assert config.hidden_size % config.num_hidden_layers == 0, (
            "Hidden size should be divisible by head numbers!"
        )
        self.attention_head_size = config.hidden_size / config.num_hidden_layers

    # TODO: Implement Positional Encoding

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_output_state: torch.Tensor | None = None,
        use_cache: bool = False,
        past_key_value: tuple | None = None,
        output_attentions: bool = False,
    ) -> dict:
        # Get Q, K and V. (b * s * h)
        query_layer = self.query(hidden_state)
        if encoder_output_state is None:
            # Self Attention, past_key_value refers to self kv cache if exists.
            key_layer = self.key(hidden_state)
            value_layer = self.value(hidden_state)
        else:
            # Cross Attention.
            # KV Cache strategy: Store and Reuse Cross KV Cache before projection and head splitting
            # If K and V are from previous, using cross attention, shape is (b * sk * h).
            if use_cache and past_key_value is not None:
                # Use Cross KV Cache
                key_layer, value_layer = past_key_value
            else:
                # Calculate Cross KV the first time.
                key_layer = self.key(encoder_output_state)
                value_layer = self.value(encoder_output_state)
            if use_cache:
                # Store Cross KV.
                key_value_cache = (key_layer, value_layer)
        # Split n-head, could work without batch dimension. (b * n * s(sk) * (h/n))
        query_layer = query_layer.view(
            *query_layer.shape[:-1], self.config.num_attention_heads, -1
        ).transpose(-3, -2)
        key_layer = key_layer.view(
            *key_layer.shape[:-1], self.config.num_attention_heads, -1
        ).transpose(-3, -2)
        value_layer = value_layer.view(
            *value_layer.shape[:-1], self.config.num_attention_heads, -1
        ).transpose(-3, -2)
        if encoder_output_state is None and use_cache:
            # Self Attention
            # KV Cache strategy: Store and Reuse Self KV Cache after head splitting.
            # Store Stripped Self Attention KV Cache. (Before concat)
            key_value_cache = (key_layer, value_layer)
            if past_key_value is not None:
                # Use Self Attention KV Cache.
                past_key, past_value = past_key_value
                key_layer = torch.cat((past_key, key_layer), dim=-2)
                value_layer = torch.cat((past_value, value_layer), dim=-2)
        # Q K^T, could work without batch dimension. (b * n * s * (h/n)) * (b * n  * (h/n) * sk) -> (b * n * s * sk)
        attention_scores = query_layer.matmul(key_layer.transpose(-2, -1))
        # div by \sqrt{d}.
        attention_scores /= self.attention_head_size**0.5
        # Use attention mask. (b * n * s * sk)
        attention_scores += attention_mask
        # Apply softmax + Dropout, scores -> probs.
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        # A * V. (b * n * s * sk) * (b * n * sk * (h/n)) -> (b * n * s * (h/n))
        hidden_state = attention_probs.matmul(value_layer)
        # Concat attentions, could work without batch dimension. (b * n * s * (h/n)) -> (b * s * h)
        hidden_state = hidden_state.transpose(-3, -2).contiguous()
        hidden_state = hidden_state.view(*hidden_state.shape[:-2], -1)
        hidden_state = self.dense(hidden_state)
        # Prepare for return values.
        return_dict = {}
        return_dict["hidden_state"] = hidden_state
        # KV Cache for self/cross attention, which depends on situations.
        if use_cache:
            return_dict["key_value_cache"] = key_value_cache
        # Return attentions
        return_dict["attention_probs"] = attention_probs if output_attentions else None
        return return_dict


class BasicFormerFeedForward(nn.Module):
    def __init__(self, config: BasicFormerConfig):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.intermediate = nn.Linear(config.hidden_size, self.intermediate_size)
        self.output = nn.Linear(self.intermediate_size, config.hidden_size)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = F.silu(self.intermediate(hidden_state))
        output_state = self.output(hidden_state)
        return output_state


class BasicFormerBlock(nn.Module):
    def __init__(
        self,
        config: BasicFormerConfig,
        has_causal_mask: bool = False,
        has_cross_attention: bool = False,
    ):
        super().__init__()
        self.config = config
        self.has_causal_mask = has_causal_mask
        self.has_cross_attention = has_cross_attention
        self.self_attention = BasicFormerAttention(config)
        self.self_dropout = nn.Dropout(config.attention_dropout_prob)
        self.self_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        if has_cross_attention:
            self.cross_attention = BasicFormerAttention(config)
            self.cross_dropout = nn.Dropout(config.attention_dropout_prob)
            self.cross_norm = nn.LayerNorm(
                config.hidden_size, eps=config.layer_norm_eps
            )
        self.output = BasicFormerFeedForward(config)
        self.output_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.full_attention_mask = None

    def apply_square_attention_mask(
        self,
        attention_mask: torch.Tensor,
        cross_attention_mask: torch.Tensor | None = None,
        has_causal_mask: bool = False,
        use_cache: bool = False,
    ) -> dict:
        r"""Get the square mask from linear mask, also get cross square mask when required, also get sliced mask for cached kv when required"""
        if cross_attention_mask is not None:
            # Save current attention mask for cross attention calculation
            cur_attention_mask = attention_mask
        if use_cache:
            # Store current mask seq_length for slicing
            cur_seq_length = attention_mask.shape[-1]
            if self.full_attention_mask is None:
                self.full_attention_mask = attention_mask
            else:
                self.full_attention_mask = torch.cat(
                    (self.full_attention_mask, attention_mask), dim=-1
                )
            attention_mask = self.full_attention_mask
        return_dict = {}
        # Set attention mask bool position
        attention_mask_bool = torch.where(
            attention_mask == self.config.is_not_pad, True, False
        )
        # Expand to (b * s * s) square bool mask
        self_mask_bool = attention_mask_bool.unsqueeze(
            -1
        ) & attention_mask_bool.unsqueeze(-2)
        # Fill blocked positions with -inf and unblocked with 0, then expand from (b * s * s) to (b * 1 * s * s)
        self_mask = torch.where(self_mask_bool, 0, -torch.inf).unsqueeze(-3)
        if cross_attention_mask is not None:
            # Set current attention mask bool position
            cur_attention_mask_bool = torch.where(
                cur_attention_mask == self.config.is_not_pad, True, False
            )
            # Set cross attention mask bool position
            cross_attention_mask_bool = torch.where(
                cross_attention_mask == self.config.is_not_pad, True, False
            )
            # Expand to (b * s * sk) square bool mask
            cross_mask_bool = cur_attention_mask_bool.unsqueeze(
                -1
            ) & cross_attention_mask_bool.unsqueeze(-2)
            # Fill blocked positions with -inf and unblocked with 0, then expand from (b * s * sk) to (b * 1 * s * sk)
            cross_mask = torch.where(cross_mask_bool, 0, -torch.inf).unsqueeze(-3)
            return_dict["cross_mask"] = cross_mask
        if has_causal_mask:
            # Append causal mask
            causal_mask = torch.full_like(self_mask, fill_value=-torch.inf).triu(1)
            self_mask += causal_mask
        if use_cache:
            # Slice the mask from (b * 1 * s * s) to (b * 1 * 1 * s)
            self_mask = self_mask[..., -cur_seq_length:, :]
        return_dict["self_mask"] = self_mask
        return return_dict

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
        cross_attention_mask: torch.Tensor | None = None,
        encoder_output_state: torch.Tensor | None = None,
        use_cache: bool = False,
        past_key_value: tuple | None = None,
        output_attentions: bool = False,
    ) -> dict:
        # Calculate attention mask
        mask_dict = self.apply_square_attention_mask(
            attention_mask=attention_mask,
            cross_attention_mask=cross_attention_mask,
            has_causal_mask=self.has_causal_mask,
            use_cache=use_cache,
        )
        # Multi-head Self Attention.
        # Order: Normalization, Dropout, Residual
        residual = hidden_state
        hidden_state = self.self_norm(hidden_state)
        # Past kv exists.
        if use_cache and past_key_value is not None:
            # Decompose cache
            if self.has_cross_attention:
                self_key_value, cross_key_value = (
                    past_key_value[:2],
                    past_key_value[2:],
                )
            else:
                self_key_value = past_key_value
        else:
            # Past kv doesn't exist, init to None
            self_key_value = None
            if self.has_cross_attention:
                cross_key_value = None
        # Self attention.
        output_dict = self.self_attention(
            hidden_state=hidden_state,
            attention_mask=mask_dict["self_mask"],
            use_cache=use_cache,
            past_key_value=self_key_value,
        )
        hidden_state = output_dict["hidden_state"]
        self_attention_probs = (
            output_dict["attention_probs"] if output_attentions else None
        )
        if use_cache:
            self_key_value = output_dict["key_value_cache"]
        hidden_state += residual
        # Cross attention
        if self.has_cross_attention:
            residual = hidden_state
            output_dict = self.cross_attention(
                hidden_state=hidden_state,
                attention_mask=mask_dict["cross_mask"],
                encoder_output_state=encoder_output_state,
                use_cache=use_cache,
                past_key_value=cross_key_value,
            )
            hidden_state = output_dict["hidden_state"]
            cross_attention_probs = (
                output_dict["attention_probs"] if output_attentions else None
            )
            if use_cache:
                cross_key_value = output_dict["key_value_cache"]
            hidden_state += residual

        # Feed Forward
        # Order: Normalization, Dropout, Residual
        residual = hidden_state
        hidden_state = self.output_norm(hidden_state)
        hidden_state = self.output(hidden_state)
        hidden_state = self.output_dropout(hidden_state)
        hidden_state += residual

        # Prepare for return values.
        return_dict = {}
        return_dict["hidden_state"] = hidden_state
        return_dict["self_attention_probs"] = (
            self_attention_probs if output_attentions else None
        )
        if use_cache:
            return_dict["key_value_cache"] = self_key_value
        if self.has_cross_attention:
            return_dict["cross_attention_probs"] = (
                cross_attention_probs if output_attentions else None
            )
            if use_cache:
                return_dict["key_value_cache"] += cross_key_value
        return return_dict


class BasicFormerPreTrainedModel(PreTrainedModel):
    config_class = BasicFormerConfig
    base_model_prefix = "basicformer"

    def __init__(self, config: BasicFormerConfig):
        super().__init__(config)
        self.config = config

    def _init_weights(self, module):
        r"""Initialize the weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # use normal dist for embeddings
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            # LayerNorm set 0 for bias, 1 for weight
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)


class BasicFormerModel(BasicFormerPreTrainedModel):
    def __init__(
        self,
        config: BasicFormerConfig,
        has_causal_mask: bool = False,
        has_cross_attention: bool = False,
    ):
        super().__init__(config)
        self.has_cross_attention = has_cross_attention
        self.embedding = BasicFormerEmbedding(config)
        self.layers = nn.ModuleList(
            [
                BasicFormerBlock(
                    config,
                    has_causal_mask=has_causal_mask,
                    has_cross_attention=has_cross_attention,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.post_init()

    @staticmethod
    def is_empty_cache(cache: DynamicCache | EncoderDecoderCache, idx: int) -> bool:
        r"""Detect if cached k/v in current position is None"""
        if isinstance(cache, DynamicCache):
            return None in vars(cache.layers[idx]).values()
        elif isinstance(cache, EncoderDecoderCache):
            return None in vars(cache.self_attention_cache.layers[idx]).values()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_output_state: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
        past_key_values: EncoderDecoderCache | DynamicCache | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        **kwargs,
    ) -> (
        BaseModelOutputWithPastAndCrossAttentions
        | BaseModelOutputWithPast
        | BaseModelOutputWithCrossAttentions
        | BaseModelOutput
    ):
        # Init attentions and output states
        self_attentions = () if output_attentions else None
        if self.has_cross_attention:
            cross_attentions = () if output_attentions else None
        hidden_states = () if output_hidden_states else None
        # Start forwarding
        hidden_state = self.embedding(input_ids)
        if output_hidden_states:
            hidden_states += (hidden_state,)
        for i, layer in enumerate(self.layers):
            # Obtain past kv
            if (
                use_cache
                and past_key_values is not None
                and not self.is_empty_cache(past_key_values, idx=i)
            ):
                if self.has_cross_attention:
                    past_key_value = (
                        past_key_values.self_attention_cache.layers[i].keys,
                        past_key_values.self_attention_cache.layers[i].values,
                        past_key_values.cross_attention_cache.layers[i].keys,
                        past_key_values.cross_attention_cache.layers[i].values,
                    )
                else:
                    past_key_value = (
                        past_key_values.layers[i].keys,
                        past_key_values.layers[i].values,
                    )
            else:
                past_key_value = None
            output_dict = layer(
                hidden_state,
                attention_mask=attention_mask,
                cross_attention_mask=encoder_attention_mask,
                encoder_output_state=encoder_output_state,
                use_cache=use_cache,
                past_key_value=past_key_value,
            )
            hidden_state = output_dict["hidden_state"]
            if use_cache:
                past_key_value = output_dict["key_value_cache"]
                if self.has_cross_attention:
                    # Update self kv
                    past_key_values.self_attention_cache.update(
                        key_states=past_key_value[0],
                        value_states=past_key_value[1],
                        layer_idx=i,
                    )
                    # Don't update cross kv unless None
                    if self.is_empty_cache(
                        past_key_values.cross_attention_cache, idx=i
                    ):
                        past_key_values.cross_attention_cache.update(
                            key_states=past_key_value[2],
                            value_states=past_key_value[3],
                            layer_idx=i,
                        )
                else:
                    # Update self kv
                    past_key_values.update(
                        key_states=past_key_value[0],
                        value_states=past_key_value[1],
                        layer_idx=i,
                    )
            # Gathering attentions.
            if output_attentions:
                self_attention_probs = output_dict["self_attention_probs"]
                self_attentions += (self_attention_probs,)
                if self.has_cross_attention:
                    cross_attention_probs = output_dict["cross_attention_probs"]
                    cross_attentions += cross_attention_probs
            # Gathering hidden states
            if output_hidden_states:
                hidden_states += (hidden_state,)
        # Prepare for return values in the format of BaseModelOutput. (last hidden state + hidden states + key_value_cache(optional) + self_attention_probs(optional) + cross_attention_probs(optional))
        if use_cache:
            if self.has_cross_attention:
                return BaseModelOutputWithPastAndCrossAttentions(
                    last_hidden_state=hidden_state,
                    past_key_values=past_key_values,
                    hidden_states=hidden_states,
                    attentions=self_attentions,
                    cross_attentions=cross_attentions,
                )
            else:
                return BaseModelOutputWithPast(
                    last_hidden_state=hidden_state,
                    past_key_values=past_key_values,
                    hidden_states=hidden_states,
                    attentions=self_attentions,
                )
        else:
            if self.has_cross_attention:
                return BaseModelOutputWithCrossAttentions(
                    last_hidden_state=hidden_state,
                    hidden_states=hidden_states,
                    attentions=self_attentions,
                    cross_attentions=cross_attentions,
                )
            else:
                return BaseModelOutput(
                    last_hidden_state=hidden_state,
                    hidden_states=hidden_states,
                    attentions=self_attentions,
                )


class BasicFormerForMaskedLM(BasicFormerPreTrainedModel):
    def __init__(self, config: BasicFormerConfig):
        super().__init__(config)
        self.config = config
        self.encoder = BasicFormerModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> MaskedLMOutput:
        model_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        output_state = model_outputs.last_hidden_state
        logits = self.lm_head(output_state)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=model_outputs.hidden_states,
            attentions=model_outputs.attentions,
        )


class BasicFormerForCausalLM(BasicFormerPreTrainedModel, GenerationMixin):
    def __init__(self, config: BasicFormerConfig):
        super().__init__(config)
        assert config.is_decoder, "Not a decoder-only model!"
        self.config = config
        self.decoder = BasicFormerModel(config, has_causal_mask=True)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.is_first_batch: bool = True
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool = True,
        past_key_values: DynamicCache | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        labels: torch.Tensor | None = None,
        **kwargs,
    ):
        model_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        logits = self.lm_head(model_outputs.last_hidden_state)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        if use_cache:
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=past_key_values,
                hidden_states=model_outputs.hidden_states,
                attentions=model_outputs.attentions,
            )
        else:
            return CausalLMOutput(
                loss=loss,
                logits=logits,
                hidden_states=model_outputs.hidden_states,
                attentions=model_outputs.attentions,
            )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool = False,
        past_key_values: DynamicCache | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        **kwargs,
    ) -> dict:
        # Slice the input id/mask except the first batch
        if self.is_first_batch:
            self.is_first_batch = False
        elif use_cache:
            input_ids = input_ids[:, :1]
            attention_mask = attention_mask[:, :1]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
            "past_key_values": past_key_values,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
        }


class BasicFormerForSeq2SeqLM(BasicFormerPreTrainedModel, GenerationMixin):
    def __init__(self, config: BasicFormerConfig):
        super().__init__(config)
        assert config.is_encoder_decoder, "Not a encoder-decoder model!"
        self.config = config
        self.encoder = BasicFormerModel(config)
        self.decoder = BasicFormerModel(
            config, has_causal_mask=True, has_cross_attention=True
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.is_first_batch: bool = True
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        decoder_input_ids: torch.Tensor | None = None,
        decoder_attention_mask: torch.Tensor | None = None,
        encoder_outputs: BaseModelOutput = None,
        use_cache: bool = True,
        past_key_values: EncoderDecoderCache | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> Seq2SeqLMOutput:
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_attention_mask=attention_mask,
            encoder_output_state=encoder_outputs.last_hidden_state,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )
        logits = self.lm_head(decoder_outputs.last_hidden_state)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def get_encoder(self) -> BasicFormerModel:
        return self.encoder

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_outputs: BaseModelOutput | None = None,
        use_cache: bool = False,
        past_key_values: EncoderDecoderCache | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        **kwargs,
    ) -> dict:
        if self.is_first_batch:
            self.is_first_batch = False
        elif use_cache:
            input_ids = input_ids[:, -1:]
        decoder_attention_mask = torch.full_like(
            input_ids, fill_value=self.config.is_not_pad
        )
        return {
            "attention_mask": attention_mask,
            "decoder_input_ids": input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "encoder_outputs": encoder_outputs,
            "use_cache": use_cache,
            "past_key_values": past_key_values,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
        }
