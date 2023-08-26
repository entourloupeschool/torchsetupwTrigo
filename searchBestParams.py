from makedata import make_data
from mlUtility import main, test_model, loaders_from_raw_data, plot_grid_predictions
from models.init_model import Model, CustomModel
from plotUtility import *
import torch
import datetime
import random as rdm


def set_seed(seed=42):
    rdm.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print("Seed set to {}".format(seed))

    return seed


set_seed()

time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

layers_config_list = []
for depth in [7, 9, 11]:
    for width in [13, 20]:
        l = [[width, False, 0] for _ in range(depth)]
        layers_config_list.append(l)

FLAGS = {
    'samples': 8000,
    'noise': 0.1,
    'width_samples': 10,
    'batch_size': 500,
    'epochs': 280,
    'ratioTrainTest': 0.7,
    'layers': layers_config_list,
    'lrsearch': [0.005, 0.004, 0.003],
    'do_rate_search': 0,
    'load': False,
    'output_dir': 'sumpluslinear/searchbestparams/',
}


# create a searchbestparams folder with time
FLAGS['output_dir'] = FLAGS['output_dir'] + time + '/'
os.makedirs(FLAGS['output_dir'], exist_ok=True)
# text file with parameters
with open(FLAGS['output_dir']+'parameters.txt', 'w') as f:
    for key, value in FLAGS.items():
        f.write(str(key) + ' : ' + str(value) + '\n')


# create the data
# x, y = make_data(FLAGS['samples'], FLAGS['noise'], FLAGS['width_samples'])

x_gen = torch.rand(FLAGS['samples'], 2) * \
    FLAGS['width_samples'] * 2 - FLAGS['width_samples']
x_power = torch.pow(torch.select(x_gen, 1, 0), 2).view(FLAGS['samples'], 1)
xcos = torch.cos(torch.select(x_gen, 1, 0)).view(FLAGS['samples'], 1)
xsin = torch.sin(torch.select(x_gen, 1, 1)).view(FLAGS['samples'], 1)
xtan = torch.tanh(torch.select(x_gen, 1, 0)).view(FLAGS['samples'], 1)
x = torch.cat((x_power, xcos, xsin, xtan), dim=1)

y = torch.cat((xcos, xsin), dim=1)
y = torch.prod(y, dim=1).view(FLAGS['samples'], 1)
y = torch.add(y, xtan).view(FLAGS['samples'], 1)
y = torch.pow(y, 2)
noise = torch.randn(FLAGS['samples'], 1) * FLAGS['noise']
y = torch.add(y, noise).view(FLAGS['samples'], 1)

# plot the data
fig, ax = plt.subplots()
ax.scatter(x_gen[:, 0], x_gen[:, 1], c=y)
ax.set_xlabel('xa[:,0]')
ax.set_ylabel('xa[:,1]')
ax.set_title('xa[:,0] * xa[:,1] + xb')
plt.show()
plt.savefig(FLAGS['output_dir']+'data.png')
plt.close()
# make dataloaders
train_loader, test_loader = loaders_from_raw_data(
    x, y, FLAGS['batch_size'], FLAGS['ratioTrainTest'])

# check size of first batch
for batch in train_loader:
    print("train_loader batch size: {}".format(len(batch)))
    print("train_loader batch first element size: {}".format(len(batch[0])))
    print("train_loader batch size: {}".format(batch[0].size()))
    break


criterion = torch.nn.MSELoss()

best_model = None
# how many last rows to take
last_rows = 3

# Initialize a dictionary to store test accuracies for each hyperparameter
hyperparameter_accuracies = {
    'lrsearch': [[] for _ in range(len(FLAGS['lrsearch']))],
    'layers': [[] for _ in range(len(FLAGS['layers']))],
}


# grid search
# Your loop for hyperparameter search
for lr_idx, lr in enumerate(FLAGS['lrsearch']):
    for layers_config_idx, layers_config in enumerate(FLAGS['layers']):
        print('-------------------')
        print('lr: {}, layers_config: {}'.format(
            lr, layers_config))

        # create the model
        model = CustomModel(in_channels=4, layers_config=layers_config)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        indiv_flag = {
            'samples': FLAGS['samples'],
            'noise': FLAGS['noise'],
            'width_samples': FLAGS['width_samples'],
            'batch_size': FLAGS['batch_size'],
            'epochs': FLAGS['epochs'],
            'ratioTrainTest': FLAGS['ratioTrainTest'],
            'layers': layers_config,
            'lr': lr,
            'do_rate': FLAGS['do_rate_search'],
            'load': False,
            'output_dir': FLAGS['output_dir']
        }

        main(train_loader, test_loader, model, optimizer, criterion,
             epochs=FLAGS['epochs'], output_dir=FLAGS['output_dir'], save=False, FLAGS=indiv_flag, in_search=True)

        # get last 5 model metrics from get model metrics
        model_metrics = model.get_model_metrics()

        # get last 5 test loss
        test_loss = [sublist[3] for sublist in model_metrics]

        # get mean of last 5 test loss
        test_loss_mean = sum(test_loss) / len(test_loss)

        # get last 5 test accuracy
        test_acc = [sublist[1] for sublist in model_metrics]

        # get mean of last 5 test accuracy
        test_acc_mean = sum(test_acc) / len(test_acc)

        # Append the test accuracy to the corresponding hyperparameter list
        hyperparameter_accuracies['lrsearch'][lr_idx].append(
            test_acc_mean)
        hyperparameter_accuracies['layers'][layers_config_idx].append(
            test_acc_mean)

        if best_model is None:
            best_model = model
            best_test_loss_mean = test_loss_mean
            best_test_acc_mean = test_acc_mean

        if test_acc_mean > best_test_acc_mean:
            best_model = model
            best_test_loss_mean = test_loss_mean
            best_test_acc_mean = test_acc_mean

            # text file with parameters
            with open(FLAGS['output_dir']+'best_model.txt', 'w') as f:
                f.write('lr: {}, layers_config: {}\n'.format(
                    lr, layers_config))
                f.write('test_loss_mean: {}\n'.format(test_loss_mean))
                f.write('test_acc_mean: {}\n'.format(test_acc_mean))
                f.write('at time: {}\n'.format(
                    datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))

bins = np.linspace(0, 1, 30)
# Plotting histograms for each hyperparameter
for hyperparam, accuracies in hyperparameter_accuracies.items():
    # Create histogram
    plt.figure()
    for i in range(len(accuracies)):
        label = FLAGS[hyperparam][i]
        if type(label) == list:
            label = str(len(label))+','+str(label[0][0])
        else:
            label = str(label)
        plt.hist(accuracies[i], bins=bins, alpha=0.5,
                 label=label)

    plt.title(f'Histogram of Test Accuracies for {hyperparam}')
    plt.xlabel('Test Accuracy')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.show()
    plt.savefig(FLAGS['output_dir']+'histogram_'+hyperparam+'.png')
    plt.close()
