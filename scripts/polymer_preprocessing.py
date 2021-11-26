import os, sys
project_dir = os.path.join(os.getcwd(),'..')
if project_dir not in sys.path:
    sys.path.append(project_dir)

hyspeclab_dir = os.path.join(project_dir, 'HySpecLab')
if hyspeclab_dir not in sys.path:
    sys.path.append(hyspeclab_dir)

from HySpecLab.utils import HyperSpectralCalibration as calibration
from HySpecLab.utils import median

import config
import numpy as np
import spectral.io.envi as envi

def read_data(capture_id, vnir=True):
    '''
        Return
        ------
            [Capture, White Reference, Black Reference] 
    '''

    spectrum = 'VNIR'
    if not(vnir):
        spectrum = 'NIR'

    if not os.path.exists( os.path.join(config.POLYMER_DATASET_DIR, 'Sensor_{}/{}/'.format(spectrum, capture_id)) ):
        raise ValueError('The capture ID does not exist.')

    ref_path = os.path.join(config.POLYMER_DATASET_DIR, 'Calibration/{}/'.format(spectrum))

    white = envi.open(os.path.join(ref_path, '{}whiteReference.hdr'.format(spectrum)))
    dark = envi.open(os.path.join(ref_path, '{}darkReference.hdr'.format(spectrum)))

    polymer_path = os.path.join(config.POLYMER_DATASET_DIR, 'Sensor_{}/{}/'.format(spectrum, capture_id))
    polymer = envi.open(os.path.join(polymer_path, 'raw.hdr'))

    bands = np.arange(polymer.nbands)

    return polymer.read_bands(bands), white.read_bands(bands), dark.read_bands(bands)

def processing(img, white_ref, dark_ref):
    img = calibration(img, white_ref, dark_ref)
    img = median(img, kernel_size=5)
    return (img / img.max())

def main():
    vnir_path = os.path.join(config.POLYMER_DATASET_DIR, 'Sensor_VNIR')
    vnir_samples = os.listdir(vnir_path)
    
    for sample in vnir_samples:
        print('Sample {}'.format(sample))
        img, white, dark = read_data(sample, vnir=True)
        img = processing(img, white, dark)
        save_dir = os.path.join(vnir_path, sample, 'processed/{}.npy'.format(sample))
        np.save(save_dir, img, allow_pickle=False)
        img=None

    nir_path = os.path.join(config.POLYMER_DATASET_DIR, 'Sensor_NIR')
    nir_samples = os.listdir(nir_path)
    for sample in nir_samples:
        print('Sample {}'.format(sample))
        img, white, dark = read_data(sample, vnir=False)
        img = processing(img, white, dark)
        save_dir = os.path.join(nir_path, sample, 'processed/{}.npy'.format(sample))
        np.save(save_dir, img, allow_pickle=False)
        img=None

if __name__ == '__main__':
    main()