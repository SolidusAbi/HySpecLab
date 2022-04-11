import torch
from torch import nn
import numpy as np
from HySpecLab.metrics import TotalVariationalReg
from tqdm import tqdm

class Reconstruction():
    @staticmethod
    def train(model: nn.Module, optimizer: torch.optim, loaders: list,
                n_epoch=50, criterion=nn.MSELoss(), tb_writer=None):
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device) 
        reg = TotalVariationalReg()

        train_loader, val_loader = loaders

        epoch_iterator = tqdm(
            range(n_epoch),
            leave=True,
            unit="epoch",
            postfix={"tls": "%.4f" % 1},
        )

        # Metrics
        loss_tr = []        
        step=0

        for epoch in epoch_iterator:
            model.train()
            for idx, (inputs, _) in enumerate(train_loader):
                optimizer.zero_grad()
                inputs = inputs.to(device)

                out = model(inputs)
                reg_loss = reg(out)
                loss = criterion(out, inputs) + 1e-5*reg_loss
                loss_tr.append(loss.item())

                if idx % 100 == 0:
                    epoch_iterator.set_postfix(tls="%.4f" % np.mean(loss_tr))

                    if tb_writer is not None:
                        tb_writer.add_scalar('train/loss', np.mean(loss_tr), step)
                        tb_writer.add_scalar('train/epoch', epoch, step) # Record the epoch per step
                        step+=1

                    loss_tr = []

                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            val_loss = []
            for inputs, _ in val_loader:
                with torch.no_grad():
                    inputs = inputs.to(device)
                    out = model(inputs)
                    loss = criterion(out, inputs)
                    val_loss.append(loss.item())
                    
            if tb_writer is not None:
                tb_writer.add_scalar('val/loss', np.mean(val_loss), epoch)