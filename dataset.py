import os 
import numpy as np
from scipy.io import loadmat

class Dataset:
    def __init__(self, dataset_dir):
        if not os.path.exists(dataset_dir):
            assert "Path does not exists"

        self.filenames = map(lambda x: os.path.join(dataset_dir, x), os.listdir(dataset_dir))
        self.x = None
        self.y = None
        self.label_map = dict()

        self.__process(self.filenames)

    def __process(self, filenames):
        for idx, filename in enumerate(filenames):
            mat_x = loadmat(filename)['preProcessedImage'][np.newaxis, :]
            mat_y = loadmat(filename)['groundTruthMap'][np.newaxis, :]
            if idx==0:
                self.x = mat_x
                self.y = mat_y
            else:
                self.x = np.concatenate([self.x, mat_x], axis=0)    
                self.y = np.concatenate([self.y, mat_y], axis=0)

            self.__gen_map()

    def __gen_map(self):
        # n_class = len(np.unique(self.y)) - 1 #remove 0 ('No Labeled data')
        classes = np.unique(self.y)
        classes = classes[classes != 0] # Discard class '0' (No Labeled data)

        import string
        header = list(string.ascii_lowercase[:len(classes)])
        for idx in range(len(classes)):
            self.label_map[header[idx]] = classes[idx]
    
    def get(self):
        idx = np.where(self.y!=0)
        _x = self.x[idx]
        tmp = self.y[idx]
        _y = np.empty(tmp.shape, dtype=np.string_)
        for key, value in self.label_map.items():
            idx = np.where(tmp == value)
            _y[idx] = key
        
        return _x, _y