# tale



https://github.com/user-attachments/assets/6ec11352-0c23-4d85-a5ea-3960f9d110d7



a small language model trained on [tinystories](https://huggingface.co/datasets/roneneldan/TinyStories).

tale is a ~50m parameter llama-style model built from scratch, trained on apple silicon and exported to gguf for local inference.

## model

* **architecture:** llamaforcausallm
* **parameters:** ~50m
* **tokenizer:** gpt-2 bpe
* **context:** 384 tokens
* **training data:** tinystories
* **training device:** apple silicon (mps)
* **format:** gguf
* **inference:** llama.cpp / ollama

## run

```
ollama run navthings/tale
```

or run the gguf directly with llama.cpp:

```
./llama-cli -m tale-step2000-f16.gguf -p "once upon a time "
```

## why

tale is an experiment, the evolution to lilstory

i built it because my little brother kept on asking me for a bedtime story

it's intentionally small.

## status

early release.

the current model is trained for 2,000 steps and can generate simple tinystories-style text, but it is still very much a work in progress.

## licence

mit
