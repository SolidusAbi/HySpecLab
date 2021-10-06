import os 
import numpy as np
from scipy.io import loadmat

class DermaDataset:
    '''
        DermaDataset:
            Classes:
                - 0: Benign lesion
                - 1: Malignant lesion

    '''
    def __init__(self, dataset_dir):
        ''' 
            Param:
            -----
            dataset_dir (str or list)
        '''
        if not isinstance(dataset_dir, list):
            dataset_dir = [dataset_dir]

        for dir in dataset_dir:
            if not os.path.exists(dir):
                assert "{} path does not exists".format(dir)
                return

        mat_files = []
        for _dataset_dir in dataset_dir:
            files = list(map(lambda x: os.path.join(_dataset_dir, x), os.listdir(_dataset_dir)))
            mat_files = mat_files + files

        self.__process(mat_files)

    def __process(self, filenames):
        for idx, filename in enumerate(filenames):
            mat = loadmat(filename)
            mat_x = mat['preProcessedImage']
            mat_y = mat['groundTruthMap']

            benign_idx = np.where(np.logical_and(mat_y>200, mat_y<400))
            malignant_idx = np.where(np.logical_and(mat_y>=400, mat_y<500))
            result_idx = (np.concatenate((malignant_idx[0], benign_idx[0])), np.concatenate((malignant_idx[1], benign_idx[1])))

            _y = np.zeros(malignant_idx[0].size + benign_idx[0].size, dtype=int)
            _y[:malignant_idx[0].size] = 1

            if idx==0:
                self.x = mat_x[result_idx]
                self.y = _y
            else:
                self.x = np.concatenate([self.x, mat_x[result_idx]], axis=0)    
                self.y = np.concatenate([self.y, _y], axis=0)

    def get(self):
        return self.x, self.y

# class Dataset:
#     def __init__(self, dataset_dir):
#         if not os.path.exists(dataset_dir):
#             assert "Path does not exists"

#         self.filenames = map(lambda x: os.path.join(dataset_dir, x), os.listdir(dataset_dir))
#         self.x = None
#         self.y = None
#         self.label_map = dict()

#         self.__process(self.filenames)

#     def __process(self, filenames):
#         for idx, filename in enumerate(filenames):
#             mat_x = loadmat(filename)['preProcessedImage'][np.newaxis, :]
#             mat_y = loadmat(filename)['groundTruthMap'][np.newaxis, :]
#             if idx==0:
#                 self.x = mat_x
#                 self.y = mat_y
#             else:
#                 self.x = np.concatenate([self.x, mat_x], axis=0)    
#                 self.y = np.concatenate([self.y, mat_y], axis=0)

#             self.__gen_map()

#     def __gen_map(self):
#         # n_class = len(np.unique(self.y)) - 1 #remove 0 ('No Labeled data')
#         classes = np.unique(self.y)
#         classes = classes[classes != 0] # Discard class '0' (No Labeled data)

#         import string
#         header = list(string.ascii_lowercase[:len(classes)])
#         for idx in range(len(classes)):
#             self.label_map[header[idx]] = classes[idx]
    
#     def get(self):
#         idx = np.where(self.y!=0)
#         _x = self.x[idx]
#         tmp = self.y[idx]
#         _y = np.empty(tmp.shape, dtype=np.string_)
#         for key, value in self.label_map.items():
#             idx = np.where(tmp == value)
#             _y[idx] = key
        
#         return _x, _y