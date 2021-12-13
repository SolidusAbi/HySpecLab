import numpy as np
from scipy.ndimage import median_filter
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from skopt.space import Categorical

class BoardLog():
    '''
        TODO: Documentation...
    '''
    def __init__(self, writer_path : str, space_name : list):
        self.space_name_ = space_name
        self.writer_ = SummaryWriter(writer_path)
        self.step_ = 0

    def __call__(self, result):
        if result.space.n_dims != len(self.space_name_):
            raise RuntimeError('BoardLog:space_name_ does not correspond to the space dimension.')

        for space_idx, space in enumerate(result.space):
            if isinstance(space, Categorical):
                for categories in space.categories:
                    value = int(result.x[space_idx] == categories)
                    self.writer_.add_scalar('Params/{}/{}'.format(self.space_name_[space_idx], categories), value, global_step=self.step_)
            else:
                self.writer_.add_scalar('Params/{}'.format(self.space_name_[space_idx]), result.x[space_idx], global_step=self.step_)

        self.writer_.add_scalar('Score', result.fun, global_step=self.step_)
        self.step_ += 1

        self.test = result

def HyperSpectralCalibration(hyspec_img, white_reference, dark_reference):
    return (hyspec_img - dark_reference) / (white_reference - dark_reference + 1e-8)

def apply_noise(hyspec_img):
    if len(hyspec_img.shape) != 3:
        raise ValueError('The has to be applied to an hyperspectral cube')

    noise = np.random.normal(0, 0.05, size=hyspec_img.shape)
    return hyspec_img.copy() + noise


def median(hyspec_img, kernel_size=3):
    if len(hyspec_img.shape) != 3:
        raise ValueError('The has to be applied to an hyperspectral cube')
    
    n_features = hyspec_img.shape[2]
    filtered = np.zeros(hyspec_img.shape)

    features_iterator = tqdm(
            range(n_features),
            leave=True,
            unit="Features",
        )

    for feature_idx in features_iterator:
        filtered[:,:, feature_idx] = median_filter(hyspec_img[:,:, feature_idx], size=kernel_size)

    return filtered

def plotSpectraAndMean(spectra, wv=None, figsize=(12, 6)):
    '''
        Parameters
        ----------
            spectra : array, shape (n_samples, n_features)
                spectral signals.

            wv: : array, shape (n_features)
                Wavelenght in nanometers 
    '''
    from matplotlib import pyplot as plt 
    fig, ax = plt.subplots(1,1, figsize=figsize)
    x = spectra.T ### spectra as NBands x NSamps

    wv = np.arange(len(x)) if not wv else wv
    
    mu = np.mean(x, axis=1)
    ax.plot(wv, x, 'c')
    ax.plot(wv, mu, 'r')
    ax.set_xlabel('Wavelenght (nm)')
    ax.set_ylabel('Reflectance')
    
    return fig

def fig_to_image(fig):
    import io
    from PIL import Image
    
    buf = io.BytesIO()
    fig.savefig(buf, format='jpeg')
    buf.seek(0)
    return Image.open(buf)