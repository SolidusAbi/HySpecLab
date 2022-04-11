import os
import torch
import numpy as np
from tqdm import tqdm
from scipy.io import loadmat
from torch.utils.data import Dataset


class PlasticDataset(Dataset):
    def __init__(self, dataset_dir, gt_dataset_dir):
        super(PlasticDataset, self).__init__()
        self.dataset_dir = dataset_dir
        self.files = list(map(lambda x: os.path.join(dataset_dir, x), os.listdir(dataset_dir)))

        self.files = sorted(list(map(lambda x: os.path.join(dataset_dir, x), os.listdir(dataset_dir))))
        self.gt_files = sorted(list(map(lambda x: os.path.join(gt_dataset_dir, x), os.listdir(gt_dataset_dir))))
        self.__process__()

    def __process__(self):
        dataset = []
        
        epoch_iterator = tqdm(
            range(len(self.files)),
            leave=True,
            unit="files"
        )

        for idx in epoch_iterator:
            file = self.files[idx]
            gt_data = loadmat(self.gt_files[idx])['gt'].reshape(-1)
            nir_data = loadmat(file)['calibratedImageNIR']
            nir_data = nir_data.reshape(-1, nir_data.shape[-1])

            labels = np.unique(gt_data)
            label_count = np.zeros(labels.size)
            for idx, label in enumerate(labels):
                label_count[idx] = len(gt_data[gt_data==label])
            
            n_samples = int(label_count.min()/2)

            for label in labels:
                idx = np.where(gt_data==label)[0]
                idx = np.random.choice(len(idx), n_samples, replace=False)
                dataset.append(nir_data[idx])

        self.dataset = torch.tensor(np.concatenate(dataset, axis=0), dtype=torch.float32)
    

    def __len__(self):
        return self.dataset.size(0) 
        
    def __getitem__(self, idx):
        return (self.dataset[idx], -1)