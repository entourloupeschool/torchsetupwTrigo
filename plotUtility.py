import matplotlib.pyplot as plt
import numpy as np
import os
import torch

plt.ion()

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, keep_all=False):
        self.reset()
        self.data = None
        if keep_all:
            self.data = []

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if self.data is not None:
            self.data.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
    

class TrainLossPlot(object):
    def __init__(self):
        self.loss_train = []
        self.fig = plt.figure()

    def update(self, loss_train):
        self.loss_train.append(loss_train)

    def plot(self):
        plt.figure(self.fig.number)
        plt.clf()
        plt.plot(np.array(self.loss_train))
        plt.title("Train loss / batch")
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.show()
        plt.draw_all()
        plt.pause(1e-3)

class AccLossPlot(object):
    def __init__(self):
        self.loss_train = []
        self.loss_test = []
        self.acc_train = []
        self.acc_test = []
        self.fig = plt.figure()

    def update(self, loss_train, loss_test, acc_train, acc_test):
        self.loss_train.append(loss_train)
        self.loss_test.append(loss_test)
        self.acc_train.append(acc_train)
        self.acc_test.append(acc_test)
        plt.figure(self.fig.number)
        plt.clf()
        plt.subplot(1,2,1)
        plt.plot(np.array(self.acc_train), label="acc. train")
        plt.plot(np.array(self.acc_test), label="acc. test")
        plt.title("Accuracy / epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(np.array(self.loss_train), label="loss train")
        plt.plot(np.array(self.loss_test), label="loss test")
        plt.title("Loss / epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        plt.draw_all()
        plt.pause(1e-3)
        
    # save the plot
    def save(self, filename, output_dir, format="pdf"):
        plt.figure(self.fig.number)
        plt.savefig(os.path.join(output_dir, filename + "." + format), format=format)
        
def plot_full_model_metrics(metrics_np):
    # create a figure with two subplots
    fig, ax1 = plt.subplots(2, 1, figsize=(12, 8))

    # plot accuracies
    ax1[0].plot(metrics_np[:, 0], 'b-', label='Training Accuracy')
    ax1[0].plot(metrics_np[:, 1], 'g-', label='Test Accuracy')
    ax1[0].set_xlabel('Epoch')
    ax1[0].set_ylabel('Accuracy')
    ax1[0].legend(loc='upper left')

    # plot losses
    ax1[1].plot(metrics_np[:, 2], 'b-', label='Training Loss')
    ax1[1].plot(metrics_np[:, 3], 'g-', label='Test Loss')
    ax1[1].set_xlabel('Epoch')
    ax1[1].set_ylabel('Loss')
    ax1[1].legend(loc='upper left')

    plt.tight_layout()
    plt.show()
    
def plot_full_model_metrics_comp(metrics_dict):
    # create a figure with two subplots
    fig, ax1 = plt.subplots(2, 1, figsize=(12, 8))

    # iterate over the metrics of each model
    for model_name, metrics_np in metrics_dict.items():
        # plot accuracies
        #ax1[0].plot(metrics_np[:, 0], label=f'{model_name} Train Acc')
        ax1[0].plot(metrics_np[:, 1], label=f'{model_name} Test Acc')

        # plot losses
        #ax1[1].plot(metrics_np[:, 2], label=f'{model_name} Train Loss')
        ax1[1].plot(metrics_np[:, 3], label=f'{model_name} Test Loss')

    ax1[0].set_xlabel('Epoch')
    ax1[0].set_ylabel('Accuracy')
    ax1[0].legend(loc='upper left')

    ax1[1].set_xlabel('Epoch')
    ax1[1].set_ylabel('Loss')
    ax1[1].legend(loc='upper left')

    plt.tight_layout()
    plt.show()


# a function to initiate a imshow plot
def init_imshow_plot():
    fig, ax = plt.subplots()
    return fig, ax

# def plot_grid_predictions(model, grid_size, width_samples, filename, output_dir):
#     d1 = torch.linspace(-width_samples, width_samples, grid_size)
#     d2 = torch.linspace(-width_samples, width_samples, grid_size)
    
#     grid = torch.zeros((grid_size, grid_size))

#     for i in range(grid_size):
#         c = torch.cos(d1[i])
#         t = torch.tanh(d1[i])
#         for j in range(grid_size):
#             s = torch.sin(d2[j])
#             tensor_input = torch.tensor([d1[i], d2[j], c, s, t]).view(1, 5)
#             grid[i, j] = test_model((tensor_input), model)
            
#     fig, ax = plt.subplots()
#     ax.imshow(grid)
#     ax.set_xlabel('d1')
#     ax.set_ylabel('d2')
#     ax.set_title('cos(d1) * sin(d2) + tanh(d1)')
#     plt.show()
#     plt.savefig(output_dir+filename)