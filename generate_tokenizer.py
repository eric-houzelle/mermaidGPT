from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence
from datasets import load_dataset
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

ds = load_dataset("Celiadraw/text-to-mermaid", split="train")
data = [f"{item['prompt']}<|sep|>{item['output']}" for item in ds]

with open("corpus_mermaid.txt", "w", encoding="utf-8") as f:
    for line in data:
        f.write(line.strip() + "\n")
        
tokenizer = Tokenizer(models.BPE())
tokenizer.normalizer = Sequence([NFD(), StripAccents()])
tokenizer.pre_tokenizer = ByteLevel()


trainer = trainers.BpeTrainer(
    vocab_size=8000,
    special_tokens=["<pad>", "<bos>", "<eos>", "<unk>", "<|sep|>"]
)



with open("corpus_mermaid.txt", "w", encoding="utf-8") as f:
    for line in data:
        f.write(line.strip() + "\n")

tokenizer.train(["corpus_mermaid.txt"], trainer)


tokenizer.decoder = ByteLevelDecoder()


tokenizer.save("tokenizer_mermaid.json")

