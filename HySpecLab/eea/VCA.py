from . import EEA

from sklearn.utils.validation import check_is_fitted
import scipy.linalg as splin
import numpy as np

class VCA(EEA):
    '''
        Vertex Component Analysis method to extract aset of endmembers (elementary spectrum) from a given
        hyperspectral image.

        Original code: https://github.com/Laadr/VCA

        References
        ----------
            [1] Nascimento, J. M., & Dias, J. M. (2005). Vertex component analysis: A fast algorithm to unmix 
            hyperspectral data. IEEE transactions on Geoscience and Remote Sensing, 43(4), 898-910.
    '''
    def __init__(self, n_endmembers:int, snr_input: float=0, random_state=None):
        super(VCA, self).__init__(n_endmembers)
        self.snr_input = snr_input
        self.random_state = random_state

    def fit(self, X, y=None):
        '''
            Parameters
            ----------
                X : array, shape (n_samples, n_features)
                    Matrix with dimensions N (number of pixels) x L (number of  channels). Each pixel is a
                    linear mixture of R endmembers signatures Y = M x s, where s = gamma x alfa.
                    Gamma is a illumination perturbation factor and alfa are the abundance
                    fractions of each endmember

                y : any, default=None
                    Ignored. This parameter exists only for compatibility with
                    sklearn.pipeline.Pipeline.

            Returns
            -------
            self : object
                Returns the instance itself.
        '''
        X_ = X.T
        L, N = X_.shape   # L number of bands (channels), N number of pixels

        if (self.n_endmembers<0 or self.n_endmembers>L):
            raise ValueError('Number of endmemembers parameter must be integer between 1 and L')

        X0 = X_ - np.mean(X_,axis=1,keepdims=True)  # zero-mean data 
        X0_p, Ud = self._proj_subspace(X0, self.n_endmembers)
        if self.snr_input == 0:
            # Compute SNR input using
            self.snr_input = self._snr_estimation(X_,X0_p)

        snr_thr = 15 + 10*np.log10(self.n_endmembers) # Thresold proposed in [1]
        if self.snr_input < snr_thr:
            # Projection to R-1 using zero-mean data
            d = self.n_endmembers - 1

            # The projection is already computed
            Ud = Ud[:,:d]

            Xp =  np.dot(Ud,X0_p[:d,:]) + np.mean(X_,axis=1,keepdims=True) # again in dimension L
                        
            x = X0_p[:d,:] #  x_p =  Ud.T * X0 is on a R-dim subspace
            c = np.amax(np.sum(x**2,axis=0))**0.5
            Y = np.vstack(( x, c*np.ones((1,N)) ))

        else:
            # Projection to R using original data
            d = self.n_endmembers

            X_p, Ud = self._proj_subspace(X_, d)
            Xp = np.dot(Ud,X_p[:d,:]) # again in dimension L (note that X_p has no zero mean)

            # x = np.dot(Ud.T,X_)
            u = np.mean(X_p, axis=1, keepdims=True) # equivalent to  u = Ud.T * r_m
            Y = X_p / (np.dot(u.T,X_p) + 1e-16)

        #############################################
        # VCA algorithm
        #############################################

        indice = np.zeros((self.n_endmembers),dtype=int)
        A = np.zeros((self.n_endmembers,self.n_endmembers))
        A[-1,0] = 1

        rng = np.random.RandomState(self.random_state)
        for i in range(self.n_endmembers):
            w = rng.rand(self.n_endmembers,1)
            # w = np.random.rand(self.n_endmembers,1)
            f = w - np.dot(A,np.dot(splin.pinv(A),w))
            f = f / splin.norm(f)
            
            v = np.dot(f.T,Y)

            indice[i] = np.argmax(np.absolute(v))
            A[:,i] = Y[:,indice[i]] # same as x(:,indice(i))

        Ae = Xp[:,indice]

        self.endmembers_ = np.clip(Ae, a_min=0, a_max=1) # Force to be sure that values goes from 0 to 1
        self.idx_ = indice
        
        return self
        
    def endmembers(self):
        '''
            Return the endmembers estimated from X.
        '''
        check_is_fitted(self)
        return self.endmembers_.T

    def _snr_estimation(self, X, Xp) -> float:
        '''
            SNR estimation following the equation from [1].

            Params
            ------
                X : array, shape (n_features, n_samples)
                    Original data

                Xp : array, shape (n_endmembers, n_samples)
                    Zeros-mean data projected onto p-subspace where p is the
                    number of endmembers, reduced dimension.

            Return
            ------
                SNR estimation
        '''
        L, N = X.shape          # L number of channels, N number of pixels
        p, N = Xp.shape  # p number of endmembers (reduced dimension)
        mean_values = np.mean(X,axis=1,keepdims=True)
        
        P_y = np.sum(X**2)/float(N)
        P_x = np.sum(Xp**2)/float(N) + np.sum(mean_values**2)
        return 10*np.log10( (P_x - p/L*P_y)/(P_y - P_x) )
