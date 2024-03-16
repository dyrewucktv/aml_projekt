import numpy as np
from scipy.io import arff

class DataLoader:
    def __init__(self):
        self.data = {}
        self.supported_datasets = ['banknote', 'kin8nm', 'phoneme', 'elevators', 'jm1', 'kdd_JapaneseVowels', 'mfeat-karhunen', 'mfeat-zernike', 'pc1']

    def __getitem__(self, item):
        if item not in self.supported_datasets:
            raise ValueError(f"Dataset {item} is not supported")

        if item not in self.data:
            f = open(f'data/{item}.arff', 'r', encoding='utf-8')
            data, _ = arff.loadarff(f)
            data = np.array(data.tolist(), dtype=object)
            f.close()
            x = data[:, :-1].astype(float)
            x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
            y_counts = np.unique(data[:,-1], return_counts=True)
            majority_class = y_counts[1].argmax()
            y = np.array([0 if _y == y_counts[0][majority_class] else 1 for _y in data[:,-1]], dtype=float)
            y = y[~np.isnan(x).any(axis=1)]
            x = x[~np.isnan(x).any(axis=1)]
            self.data[item] = (x, y)

        return self.data[item]
    
    def get_supported_datasets(self):
        return self.supported_datasets