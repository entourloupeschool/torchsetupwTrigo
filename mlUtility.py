# Description: Utility functions for machine learning
from tqdm import tqdm
import datetime
import matplotlib.pyplot as plt
import torch
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torch.utils.data import TensorDataset, DataLoader
from plotUtility import *

# epoch function


def epoch(data, model, criterion, optimizer=None, cuda=False):
    """
    Make a pass (called epoch in English) on the data `data` with the
     model `model`. Evaluates `criterion` as loss.
     If `optimizer` is given, perform a training epoch using
     the given optimizer, otherwise, perform an evaluation epoch (no backward)
     of the model.
    """

    # indicates whether the model is in eval or train mode (some layers behave differently in train and eval)
    model.eval() if optimizer is None else model.train()

    lt_acc = []
    lt_loss = []

    # we iterate on the batches
    for i, (input, target) in enumerate(tqdm(data)):

        if cuda:  # only with GPU, and not with CPU
            input = input.cuda()
            target = target.cuda()

        # forward
        output = model(input)
        loss = criterion(output, target)

        lt_loss.append(loss.item())

        # backward if we are training
        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # compute accuracy
        # accuracy from the output and the target
        # if output is within 0.05 of the target, then it is correct
        # print(output.size(), output)
        # print(target.size(), target)
        accuracy = torch.abs(output - target) < 0.1
        accuracy = torch.sum(accuracy).float() / accuracy.numel()
        lt_acc.append(accuracy.item())

    # return the mean loss and mean accuracy
    return sum(lt_loss)/len(lt_loss), sum(lt_acc)/len(lt_acc)


def main(train, test, model, optimizer, criterion, epochs=5, cuda=False, output_dir='', scheduler=False, save=True, writer=None, FLAGS=None, in_search=False):

    if scheduler:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.6)

    if cuda:  # only with GPU, and not with CPU
        cudnn.benchmark = True
        model = model.cuda()
        criterion = criterion.cuda()

    time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    file_name = "model_" + str(time)

    # folder for model and results
    if output_dir and not in_search:
        output_dir = output_dir + file_name + '/'
        os.makedirs(output_dir, exist_ok=True)
        # text file with parameters
        with open(output_dir+'parameters.txt', 'w') as f:
            for key, value in FLAGS.items():
                f.write(str(key) + ' : ' + str(value) + '\n')

        print("Results will be saved to", output_dir)

    # init imshow plot
    fig, ax = plot_grid_predictions(
        model, 100, FLAGS['width_samples'], 'grid_predictions.png')

    # We iterate on the epochs
    for i in range(epochs):
        loss_train, acc_train = epoch(train, model, criterion, optimizer, cuda)

        if writer:
            writer.add_scalar('Loss/train', loss_train, i)
            writer.add_scalar('Accuracy/train', acc_train, i)

        if scheduler:
            # Update learning rate
            scheduler.step()

        loss_test, acc_test = epoch(test, model, criterion, cuda=cuda)
        if writer:
            writer.add_scalar('Loss/test', loss_test, i)
            writer.add_scalar('Accuracy/test', acc_test, i)

        model.add_epoch_metrics([round(acc_train, 2), round(
            acc_test, 2), round(loss_train, 3), round(loss_test, 3)])

        print(
            f"********** EPOCH {i+1} acc train={acc_train:.2f}%, acc test={acc_test:.2f}%, loss train={loss_train:.3f}, loss test={loss_test:.3f} **********")

        # ten times in the total number of epochs
        if i % 10 == 0 and not in_search:
            # update grid predictions
            plot_grid_predictions(
                model, 100, FLAGS['width_samples'], 'grid_predictions@epoch'+str(i)+'.png', output_dir=output_dir, fig=fig, ax=ax)

            with open(output_dir+'results.txt', 'a') as f:
                # go to the bottom of the text and add a new line
                f.write('\n')
                # write the results of the epoch
                f.write(
                    f"********** EPOCH {i} acc train={acc_train:.2f}%, acc test={acc_test:.2f}%, loss train={loss_train:.3f}, loss test={loss_test:.3f} **********")
                # go to the bottom of the text and add a new line
                f.write('\n')

        if save:
            # save model if the test accuracy has increased
            if i > 3:
                model.save(model, file_name, output_dir, i, optimizer)
                print("Model saved to", output_dir+file_name+".pth")

    if not in_search:
        plot_grid_predictions(
            model, 100, FLAGS['width_samples'], 'grid_predictions@final.png', output_dir=output_dir, fig=fig, ax=ax)

        with open(output_dir+'results.txt', 'a') as f:
            # write the results of the epoch
            f.write(
                f"********** FINAL EPOCH acc train={acc_train:.2f}%, acc test={acc_test:.2f}%, loss train={loss_train:.3f}, loss test={loss_test:.3f} **********")

    print('Finished Training')

    if writer:
        writer.close()
        writer.flush()

    print('len of model metrics', len(model.get_model_metrics()))
    print('len of last element of model metrics', len(
        model.get_model_metrics()[-1]))


def test_model(input_tensor, model):
    with torch.no_grad():  # Ensure no gradients are computed during this operation
        predicted_tensor = model(input_tensor)

    predictions = predicted_tensor.item()

    return predictions


def loaders_from_raw_data(x, y, batch_size, ratioTrainTest=0.8):
    trainNumSamples = int(x.shape[0] * ratioTrainTest)

    train_set = TensorDataset(x[:trainNumSamples], y[:trainNumSamples])
    test_set = TensorDataset(x[trainNumSamples:], y[trainNumSamples:])

    train_loader = DataLoader(train_set, batch_size=batch_size)
    target_loader = DataLoader(test_set, batch_size=batch_size)

    return train_loader, target_loader


def plot_grid_predictions(model, grid_size, width_samples, filename, output_dir=None, fig=None, ax=None):
    d1 = torch.linspace(-width_samples, width_samples, grid_size)
    d2 = torch.linspace(-width_samples, width_samples, grid_size)

    grid = torch.zeros((grid_size, grid_size))

    for i in range(grid_size):
        x = torch.pow(d1[i], 2)
        c = torch.cos(d1[i])
        t = torch.tanh(d1[i])
        for j in range(grid_size):
            s = torch.sin(d2[j])
            tensor_input = torch.tensor([x, c, s, t]).view(1, 4)
            grid[i, j] = test_model((tensor_input), model)

    # rotation of the grid
    grid = torch.rot90(grid, 1, [0, 1])

    if fig:
        ax.imshow(grid)
        plt.show()
        if output_dir:
            plt.savefig(output_dir+filename)
        plt.close()

        return fig, ax
    else:
        fig, ax = plt.subplots()
        ax.imshow(grid)
        ax.set_xlabel('d1')
        ax.set_ylabel('d2')
        ax.set_title('cos(d1) * sin(d2) + tanh(d1)')
        plt.show()
        if output_dir:
            plt.savefig(output_dir+filename)
        plt.close()

        return fig, ax
