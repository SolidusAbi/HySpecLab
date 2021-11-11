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