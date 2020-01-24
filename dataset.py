from PIL import Image

import torch
from torch.utils.data import Dataset
from pathlib import Path

class TextDataset(Dataset):
    
    def __init__(self, data_path, split_file, transform=None):
        self.data_path = Path(data_path)
        self.split_file = split_file
        self.transform = transform

        self.images = []
        self.labels = []
        with open(self.data_path / 'labels' / split_file, 'r') as f:
            labels_file = f.read().splitlines()
        for line in labels_file:
            line = line.split(' ')
            self.images.append(self.data_path / 'images' / line[0])
            self.labels.append(int(line[1]))
        self.cls_to_int = {}
        self.int_to_cls = {}
        with open('classes.txt', 'r') as f:
            for i, line in enumerate(f.read().splitlines()):
                self.cls_to_int[line] = i
                self.int_to_cls[i] = line

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        label = self.labels[idx]

        image.show()
        if self.transform is not None:
            image = self.transform(image)

        return image, label  


if __name__=='__main__':
    dataset = TextDataset('./data', 'val.txt')
    print(dataset[0])
