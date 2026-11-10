# Tiny_Shakespeare
This is a decoder-only Transformer on the classic Tiny Shakespeare dataset for educational purposes.

Tiny Shakespeare is a GPT style LLM built only for learning purposes. The main goal is to understand the Transformer architecture and Atrtention mechanism from scratch. It is built on the Bigram language model.


This is a decoder-only transformer. It has no encoder or cross-attention head. It has self-attention and feed-forward layers. That is why this model generates text unconditionally according to a given dataset. It has an autoregressive property to sample from the dataset it was trained on.
