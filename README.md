# BasicFormer, a Huggingface-compatible Transformer written in pytorch

## 📖Introduction

The purpose of writing this BasicTransformer is, apparently, to serve as a bridge for NLP newcomers, so that nobody has to write a Hugging Face-compatible Transformer from scratch, which is a truly mentally tiresome task. 😫

🥳Currently it implements the following features:
- [x] Encoder-only Transformer
- [x] Decoder-only Transformer
- [x] Encoder-Decoder Transformer with Cross Attention Mask
- [x] Key/Value Cache
- [x] Auto mask handling
- [x] `GenerationMixin` compatible for official generation procedure.

😓It lacks the following features:
- [ ] Positional Encoding (wont fix, there are many implementations, better DIY.)
- [ ] Tied weights between the input embedding layer and language model head.

## 💻Backgrounds

As the writer of this Transformer, I am not a specialist in NLP field, just a beginner. I acknowledged that currently `transformers` library from Huggingface holds the biggest ecosystem and almost all NLP researchers use it for research coding. So I decided to build a transformer model from scratch, while incorporating it into the huggingface ecosystem, utilizing the `generate()` method seamlessly, and equipping with "Key Value Cache" for faster and memory-friendly generation.

Currently I've finished the complete backbones. It contains 4 major models: `BasicFormerModel`, `BasicFormerForMaskedLM`, `BasicFormerForCausalLM` and `BasicFormerForSeq2SeqLM`, among them, `BasicFormerModel` is a complete Transformer that contains a full working transformer without language mission head, it can be switched from encoder to decoder or vice-versa with the parameter `has_causal_mask` and `has_cross_attention`, while other three models are task-specific models.

`BasicFormerForMaskedLM` is an encoder-only model. It can be used for MLM tasks. It doesn't support the `generate()` function due to its I/O property.

`BasicFormerForCausalLM` is a decoder-only model. It can be used on CLM task. It can call `generate()` function to directly generate the outputs.

`BasicFormerForSeq2SeqLM` is a encoder-decoder model. It can be used on Seq2Seq task. It can call `generate()` function to directly generate the outputs.

During the writing of these models, I've encountered many problems, including but not limited to:
- How to apply cross attention mask?
- To satisfy task-specific outputs: Which level of module to stack the output attentions? Which level of module to stack the hidden states?
- How & When to store self key/value cache?
- How & When to store cross key/value cache?
- When using key/value cache, how to slice the mask so that the mask shape won't mismatch with the shape of attention score?
- Given the `update()` stragety of `DynamicCache`: When to update self key/value cache? When to update cross key/value cache?

The answers apparently lie within the source code. Don't be afraid to read it—it's probably the most beginner-friendly Hugging Face-compatible model to read, thanks to abundant comments. 🤗

# 🚀Usage

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and a compatible [python](https://docs.astral.sh/uv/guides/install-python/). Version 3.12 has been tested; anything ≥3.10 should work. You can adjust `requires-python` in `pyproject.toml` and the `.python-version` file accordingly.

Clone this repo, then:
```bash
uv sync
```

To test all models at once, run:
```bash
uv run tests/generate-raw.py
```
