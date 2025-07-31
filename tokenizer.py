from tokenizers import Tokenizer
import torch 


def load_tokenizer(path):
    tokenizer = Tokenizer.from_file(path)
    tokenizer.enable_padding(pad_id=0, pad_token="<pad>")
    tokenizer.enable_truncation(max_length=128)

    def encode(text):
        return torch.tensor(tokenizer.encode(text).ids)

    def decode(tokens):
        return tokenizer.decode(tokens)

    vocab = tokenizer.get_vocab()
    stoi = {k: v for k, v in vocab.items()}
    itos = {v: k for k, v in vocab.items()}
    pad_token_id = stoi["<pad>"]
    eos_token_id = stoi.get("<eos>")
    return stoi, itos, encode, decode, pad_token_id, eos_token_id
