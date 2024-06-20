import h5py
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os

class EmbeddingDataset(Dataset):
    def __init__(self, h5_dir, df, train=True):
        self.df = df
        # self.embeddings = None
        self.train = train
        self.id = self.df['folder_id'].to_list()
        self.path = [os.path.join(h5_dir, i+'.h5') for i in self.id]
        self.labels = self.df['her2'].to_list()
        self.embeddings = []

        for p in self.path:
            with h5py.File(p, 'r') as f:
                self.embeddings.append(f['features'][:])
                # self.labels = f['labels'][:]

        # split_idx = int(0.8 * len(self.embeddings))  # 80-20 train-validation split
        # if train:
        #     self.embeddings = self.embeddings[:split_idx]
        #     self.labels = self.labels[:split_idx]
        # else:
        #     self.embeddings = self.embeddings[split_idx:]
        #     self.labels = self.labels[split_idx:]

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = self.labels[idx]
        return torch.tensor(embedding, dtype=torch.float32), torch.tensor(label, dtype=torch.long)