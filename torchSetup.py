import torch
import random as rdm
import numpy as np
import multiprocessing
from mlUtility import main, loaders_from_raw_data
from models.init_model import Model
from makedata import *
from plotUtility import *
import datetime

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/experiments")

def set_seed(seed=42):
    rdm.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print("Seed set to {}".format(seed))

    return seed


set_seed()

num_cores = multiprocessing.cpu_count()
print("This machine has {} CPU cores.".format(num_cores))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print("PyTorch Version: ", torch.__version__)

time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

FLAGS = {
    'samples': 5000,
    'noise': 0.1,
    'width_samples': 10,
    'batch_size': 50,
    'epochs': 100,
    'ratioTrainTest': 0.7,
    'neuronsf1': 6,
    'neuronsf2': 8,
    'lr': 0.003,
    'load': False,  # 'sumpluslinear/models/model_2023-08-15_09:06:20.pth', # false or 'path/to/model.pth'
    'output_dir': 'sumpluslinear/results/',
}

# create the data
# x, y = make_data(FLAGS['samples'], FLAGS['noise'], FLAGS['width_samples'])

x_gen = torch.rand(FLAGS['samples'], 2) * \
    FLAGS['width_samples'] * 2 - FLAGS['width_samples']
xcos = torch.cos(torch.select(x_gen, 1, 0)).view(FLAGS['samples'], 1)
xsin = torch.sin(torch.select(x_gen, 1, 1)).view(FLAGS['samples'], 1)
xtan = torch.tanh(torch.select(x_gen, 1, 0)).view(FLAGS['samples'], 1)
x = torch.cat((xcos, xsin, xtan), dim=1)

y = torch.cat((xcos, xsin), dim=1)
y = torch.prod(y, dim=1).view(FLAGS['samples'], 1)
y = torch.add(y, xtan).view(FLAGS['samples'], 1)
noise = torch.randn(FLAGS['samples'], 1) * FLAGS['noise']
y = torch.add(y, noise).view(FLAGS['samples'], 1)

# plot the data
fig, ax = plt.subplots()
ax.scatter(x_gen[:, 0], x_gen[:, 1], c=y)
ax.set_xlabel('xa[:,0]')
ax.set_ylabel('xa[:,1]')
ax.set_title('xa[:,0] * xa[:,1] + xb')
plt.show()
# plt.savefig('sumpluslinear/models/data'+str(time)+'.png')

# model
model = Model(in_channels=3, f1_width=FLAGS['neuronsf1'], f2_width=FLAGS['neuronsf2'],
              f3_width=FLAGS['neuronsf2'], f4_width=FLAGS['neuronsf2'], f5_width=FLAGS['neuronsf2'])

if FLAGS['load']:
    loaded = torch.load(FLAGS['load'])
    model.load_state_dict(loaded['model_state_dict'])

optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS['lr'])
criterion = torch.nn.MSELoss()

train_loader, target_loader = loaders_from_raw_data(
    x, y, FLAGS['batch_size'], FLAGS['ratioTrainTest'])

# check size
print("train_loader size: {}".format(len(train_loader)))
print("target_loader size: {}".format(len(target_loader)))

# check size of first batch
for batch in train_loader:
    print("train_loader batch size: {}".format(len(batch)))
    print("train_loader batch first element size: {}".format(len(batch[0])))
    print("train_loader batch size: {}".format(batch[0].size()))
    break

# training
main(train_loader, target_loader, model, optimizer, criterion,
     epochs=FLAGS['epochs'], cuda=False, output_dir=FLAGS['output_dir'], scheduler=False, save=False, writer=writer, FLAGS=FLAGS)
