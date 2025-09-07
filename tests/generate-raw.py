from transformers import AutoTokenizer

from basicformer.config_basicformer import BasicFormerConfig
from basicformer.modeling_basicformer import (
    BasicFormerForCausalLM,
    BasicFormerForMaskedLM,
    BasicFormerForSeq2SeqLM,
)

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
tokenizer.padding_side = "left"

inputs = tokenizer(
    ["Nice to see you here!", "This is a test."],
    return_tensors="pt",
    padding=True,
)

print("Start test for MLM")

config = BasicFormerConfig(
    vocab_size=tokenizer.vocab_size,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
)

model = BasicFormerForMaskedLM(config)
print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

print(f"Input shape: {inputs['input_ids'].shape}")
print(f"Input mask shape: {inputs['attention_mask'].shape}")

outputs = model(**inputs)

print(f"Output logits shape: {outputs.logits.shape}")

print("Start test for CLM.")

config = BasicFormerConfig(
    vocab_size=tokenizer.vocab_size,
    is_decoder=True,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
)

model = BasicFormerForCausalLM(config)
print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

print("Without cache.")
outputs = model.generate(**inputs, use_cache=False, max_new_tokens=20)

print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

print("With cache.")
outputs = model.generate(**inputs, max_new_tokens=20)

print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

print("Start test for Seq2SeqLM.")

config = BasicFormerConfig(
    vocab_size=tokenizer.vocab_size,
    is_encoder_decoder=True,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
)

model = BasicFormerForSeq2SeqLM(config)
print(f"parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

print("Without cache.")
outputs = model.generate(**inputs, use_cache=False, max_new_tokens=20)

print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

print("With cache.")
outputs = model.generate(**inputs, max_new_tokens=20)

print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
