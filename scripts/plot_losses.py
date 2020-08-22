import keras
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output


class PlotLosses(keras.callbacks.Callback):
    def __init__(self, printInterval=5, name = 'mean_absolute_percentage_error'):
        self.initVars()
        self.printInterval = printInterval
        self.name = name

    def initVars(self):
        self.i = 0
        self.x = []

        self.losses = []
        self.mse = []
        self.mae = []

        self.val_mse = []
        self.val_mae = []
        self.val_loss = []
        self.fig = plt.figure()

    def on_epoch_end(self, epoch, logs={}):
        self.i += 1

        self.losses.append(logs.get('loss'))
        self.mse.append(logs.get('mse'))
        self.mae.append(logs.get(self.name))
        self.val_mse.append(logs.get('val_mse'))
        self.val_mae.append(logs.get(f'val_{self.name}'))
        self.val_loss.append(logs.get('val_loss'))

        self.x.append(self.i)
        if self.i % self.printInterval != 0:
            return
        plt.figure(figsize=(5, 5))
        clear_output(wait=True)
        plt.figure(figsize=(16, 4))
        plt.subplot(131)
        plt.title('Loss plot')
        plt.xlabel('Number of epochs')
        plt.ylabel('Loss value')
        render_idx = np.arange(len(self.x))
        if len(render_idx) > 50:
            render_idx = render_idx[-50:]
        plt.plot(np.array(self.x)[render_idx],
                 np.array(self.losses)[render_idx],
                 label="train loss")
        plt.plot(np.array(self.x)[render_idx],
                 np.array(self.val_loss)[render_idx],
                 label="val loss")
        plt.legend()

        plt.subplot(132)
        plt.title('mse')
        plt.xlabel('Number of epochs')
        plt.plot(np.array(self.x)[render_idx],
                 np.array(self.mse)[render_idx],
                 label="train mse")
        plt.plot(np.array(self.x)[render_idx],
                 np.array(self.val_mse)[render_idx],
                 label="val mse")
        plt.legend()

        plt.subplot(133)
        plt.title(self.name)
        plt.plot(np.array(self.x)[render_idx],
                 np.array(self.mae)[render_idx],
                 label="train ")
        plt.plot(np.array(self.x)[render_idx],
                 np.array(self.val_mae)[render_idx],
                 label="val ")
        plt.legend()
        plt.tight_layout()
        plt.show()
