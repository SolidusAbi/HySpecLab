from . import EEA

class SVDD(EEA):
    '''
        Support Vector Data Descriptor (SVDD) to extract a set of endmembers 
        (elementary spectrum) from a given hyperspectral image.

        This implementation uses Radial Basis Function (RBF) as kernel in SVDD. 
        The sigma controls how tightly the SVDD models the support boundaries.
    '''

    def __init__(self, sigma: float, d:int, C=100.0) -> None:
        '''
            Parameters
            ----------
                sigma: float
                    The parameter sigma for RBF kernel. This parameters controls 
                    how tightly the SVDD models the support boundaries.
                d:

                C: float

                
        '''
        super(SVDD, self).__init__(0)
        self.sigma
        self.d = d
        self.C