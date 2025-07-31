from torch.utils.data import Dataset
import torch

    
class MermaidDataset(torch.utils.data.Dataset):
    def __init__(self, list_of_lines, block_size, encode_fn):
        self.data = list_of_lines
        self.block_size = block_size
        self.encode = encode_fn

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        ids = self.encode(text)
        ids = ids[:self.block_size]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        return x, y
