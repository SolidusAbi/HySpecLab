import os 
import numpy as np
import pandas as pd
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
            # mat_x = mat['calibratedHsCube']
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

    def get(self, dataframe=False):
        if dataframe:
            return pd.DataFrame(self.x, columns=list(
                    map(lambda x: "Band {}".format(x), np.arange(1, self.x.shape[1]+1))
                )), pd.DataFrame(self.y, columns=['Target'])
            

        return self.x, self.y