import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule


class BaseModel(LightningModule):
    """ Basic neural network class to implement methods all havin in common. """

    def __init__(self, hparams, dataset):
        """ Initialize dataset, data loader, network dimensions, and torch layers. """
        super().__init__()
        self.dataset = dataset
        self.save_hyperparameters(hparams)
        self._build()

    def configure_optimizers(self):
        """ Configure optimizers and scheduler to return for pytorch_lightning. """
        optimizer = getattr(torch.optim, self.hparams.optimizer)(
            self.parameters(),
            **(self.hparams.optimizer_params)
        )

        if self.hparams.scheduler is None:
            return optimizer
        else:
            scheduler = getattr(torch.optim.lr_scheduler, self.hparams.scheduler)(
                optimizer,
                **(self.hparams.scheduler_params)
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler,
                'monitor': self.hparams.monitor
            }

    def training_step(self, batch, batch_idx):
        """ Perform training stel given a batch. """
        x, y = batch
        y_hat = self(x)  # self.forward(..)

        y = y.flatten()
        y_hat = y_hat.flatten()

        # return loss
        return {
            'loss': F.l1_loss(y_hat, y)
        }

    def training_epoch_end(self, outputs):
        """ Calculate loss mean after epoch and for logging/evaluation. """
        loss_mean = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('loss', loss_mean)

    def validation_step(self, batch, batch_idx):
        """ Perform validation step given a validation batch. """
        x, y = batch
        y_hat = self(x)  # self.forward(..)

        y = y.flatten()
        y_hat = y_hat.flatten()

        # calculate and return y, y_hat, and loss
        # for later evaluation in validation_epoch_end
        return {
            'y': y,
            'y_hat': y_hat,
            'val_loss': F.l1_loss(y_hat, y)
        }

    def validation_epoch_end(self, outputs):
        """ Calculate suitable metrics on validation set for logging and evaulation. """
        y = torch.cat([x['y'] for x in outputs])
        y_hat = torch.cat([x['y_hat'] for x in outputs])
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()

        # calculate loss
        metric_dict = {}
        metric_dict[f'val/mae'] = F.l1_loss(y_hat, y)
        metric_dict[f'val/mse'] = F.mse_loss(y_hat, y)

        for key in metric_dict:
            self.log(key, metric_dict[key])
        self.log('val_loss', val_loss_mean)

    def test_step(self, batch, batch_idx):
        """ Perform test step given a validation batch. """
        x, y = batch
        y_hat = self(x)  # self.forward(..)

        y = y.flatten()
        y_hat = y_hat.flatten()

        # calculate and return y, y_hat, and loss
        # for later evaluation in validation_epoch_end
        return {
            'y': y,
            'y_hat': y_hat,
            'test_loss': F.l1_loss(y_hat, y)
        }

    def test_epoch_end(self, outputs):
        """ Calculate suitable metrics on test set for logging and evaulation. """
        y = torch.cat([x['y'] for x in outputs])
        y_hat = torch.cat([x['y_hat'] for x in outputs])
        val_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()

        # calculate loss
        metric_dict = {}
        metric_dict[f'test/mae'] = F.l1_loss(y_hat, y)
        metric_dict[f'test/mse'] = F.mse_loss(y_hat, y)

        for key in metric_dict:
            self.log(key, metric_dict[key])
        self.log('test_loss', val_loss_mean)
