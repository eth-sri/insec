from transformers import AutoTokenizer, AutoModelForCausalLM


for model in [
    "bigcode/starcoderbase-3b",
    "codellama/CodeLlama-7b-hf",
    "bigcode/starcoder2-3b",
    "bigcode/starcoder2-7b",
    "bigcode/starcoder2-15b",
]:
    tokenizer = AutoTokenizer.from_pretrained(model)
    m = AutoModelForCausalLM.from_pretrained(model)
