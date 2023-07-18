import torch
from torch.utils.data import Dataset
from ImageGenerator import ImageGenerator
from JitterFilter import JitterFilter

class JitteredDataset(Dataset):
    def __init__(self, N, maxJitter, length=100):
        self.N = N
        self.length = length
        self.maxJitter = maxJitter
        self.Generator = ImageGenerator(self.N)
        self.Filter = JitterFilter()

    def __len__(self):
        return self.length

    def __get_item__(self):
        groundTruthNumpy = self.Generator.genericNoise()
        jitteredTruthNumpy = self.Filter.rowJitter(groundTruthNumpy, self.N,
                                                   self.maxJitter)

        groundTruthTorch = torch.from_numpy(groundTruthNumpy).type(torch.float32) 
        jitteredTruthTorch = torch.from_numpy(jitteredTruthNumpy).type(torch.float32)

        return groundTruthTorch, jitteredTruthTorch

if __name__ == "__main__":
    dataset = JitteredDataset(5, 2)
    print(dataset)
