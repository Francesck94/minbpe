# minbpe (fork)

My implementation of the BPE (Byte Pair Encoding) tokenizers from Andrej Karpathy's [minbpe](https://github.com/karpathy/minbpe) repository, following his [YouTube lecture](https://www.youtube.com/watch?v=zduSFxRajkE) and the exercises in [exercise.md](exercise.md).

## What is this?

This is a fork of [karpathy/minbpe](https://github.com/karpathy/minbpe). Instead of using the reference solutions, I'm implementing each tokenizer from scratch by following the lecture and the step-by-step exercise progression. The goal is to deeply understand how BPE tokenization works in modern LLMs (GPT-2, GPT-4, Llama, etc.).

## Progress

| Exercise | Description | Status |
|----------|-------------|--------|
| Step 1 | `BasicTokenizer` — train, encode, decode | Done |
| Step 2 | `RegexTokenizer` — regex pre-splitting (GPT-4 pattern) + special tokens | Done |
| Step 3 | `GPT4Tokenizer` — load GPT-4 merges and match tiktoken output | Not started |
| Step 4 | Special token handling (match tiktoken with `allowed_special`) | Not started |
| Step 5 | Explore sentencepiece / Unicode code point BPE (Llama-style) | Not started |

## Repo structure

```
minbpe/
  base.py          # Base Tokenizer class, helpers (get_stats, merge, save/load)
  basic.py         # BasicTokenizer — BPE directly on raw UTF-8 bytes
  regex.py         # RegexTokenizer — regex pre-split + BPE per chunk + special tokens
notebooks/         # Jupyter notebooks used for experimentation
tests/             # Tests and sample data (taylorswift.txt)
train.py           # Script to train tokenizers and save vocab to disk
exercise.md        # Step-by-step exercises from Karpathy
lecture.md         # Full lecture transcript
```

## Quick start

```python
# Basic tokenizer (no regex splitting)
from minbpe import BasicTokenizer
tokenizer = BasicTokenizer()
tokenizer.train("aaabdaaabac", vocab_size=256 + 3)
print(tokenizer.encode("aaabdaaabac"))  # [258, 100, 258, 97, 99]
print(tokenizer.decode([258, 100, 258, 97, 99]))  # aaabdaaabac

# Regex tokenizer (GPT-4 style splitting)
from minbpe import RegexTokenizer
tokenizer = RegexTokenizer()
tokenizer.train(training_text, vocab_size=512)
tokens = tokenizer.encode("Hello world!!!!?")
print(tokenizer.decode(tokens))  # Hello world!!!!?
```

## Tests

```bash
pip install pytest
pytest -v .
```

## References

- Original repo: [karpathy/minbpe](https://github.com/karpathy/minbpe)
- Lecture video: [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)
- GPT-2 paper: [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [tiktoken](https://github.com/openai/tiktoken) (OpenAI's tokenizer library)

## License

MIT
