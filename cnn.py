# Logging
import argparse
import logging
import sys

# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.models as models


# Basics
import matplotlib.pyplot as plt
import numpy as np

# Custom files
from baseline_cnn import Nnet, Loader, ExNet, TronNet
from utils import evaluate, weights_init, get_k_fold_indecies, get_transformers, get_current_time, sklearn_metrics, sklearn_acc_per_class


TIME = None

NETS = {
    "ExNet": ExNet,
    "Nnet": Nnet,
    "TransferNet": None,
    "TronNet": TronNet
}
settings = {
    'EPOCHS': 50,
    'BATCH_SIZE': 256,
    'LR': 0.008,
    'DECAY': 0,
    'NUM_CLASSES': 201,
    'RANDOM_SEED': 42,
    'K-FOLD': False,
    'WLOSS': True,
    'K-FOLD-NUMBER': 2,
    'NNET': None,
    'TRANSFORMER': "default",
    'DATA_PATHS': {
        'TRAIN_CSV': 'train.csv',
        'TEST_CSV': 'test.csv',
        'DATASET_PATH': './datasets/cs154-fa19-public/'
    }
}
train_epoch_losses = []
val_epoch_losses = []
logging.basicConfig(filename='app.log', filemode='w', format='%(asctime)s %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def check_cuda():
    # Check if your system supports CUDA
    use_cuda = torch.cuda.is_available()

    # Setup GPU optimization if CUDA is supported
    if use_cuda:
        computing_device = torch.device("cuda")
        extras = {"num_workers": 1, "pin_memory": True}
        logging.info("CUDA is supported")
    else:  # Otherwise, train on the CPU
        computing_device = torch.device("cpu")
        extras = False
        logging.info("CUDA NOT supported")
    return computing_device, extras


def train(dataset):
    computing_device, extra = check_cuda()

    # Save all the k models to compare
    nnets = []
    batch_size = settings['BATCH_SIZE']
    # Get a lists of train-val-split for k folds

    if settings['K-FOLD']:
        indices = get_k_fold_indecies(dataset, settings['RANDOM_SEED'], k=settings['K-FOLD-NUMBER'])
    else:
        indices = list(range(len(dataset)))
        validation_split = .2
        split = int(np.floor(validation_split * len(dataset)))
        np.random.seed(settings['RANDOM_SEED'])
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        indices = [(train_indices, val_indices)]

    for k, (train_indices, val_indices) in enumerate(indices):
        logging.info("#" * 20)
        logging.info("Training Model {}".format(k))

        # Load data for this fold
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=10)
        validation_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=10)

        # Initialize CNN
        if settings['NNET'] is None:
            net = models.resnet18(pretrained=True)

            # Freeze parameters, so gradient not computed here
            for param in net.parameters():
                param.requires_grad = False

            num_ftrs = net.fc.in_features
            net.fc = nn.Linear(num_ftrs, settings['NUM_CLASSES'])
            net = net.to(computing_device)
        else:
            net = settings['NNET'](settings['NUM_CLASSES']).to(computing_device)
            net.apply(weights_init)

        # Initialize optimizer and criterion
        if settings["WLOSS"]:
            criterion = nn.CrossEntropyLoss(weight=dataset.get_class_weights().to(computing_device))
        else:
            criterion = nn.CrossEntropyLoss()

        if net.__class__.__name__ == "TransferNet":

            optimizer = optim.Adam(net.main.classifier.parameters(), lr=settings["LR"], weight_decay=settings["DECAY"])
        else:
            optimizer = optim.Adam(net.parameters(), lr=settings["LR"], weight_decay=settings["DECAY"])

        # Fit and save model to file
        if settings['K-FOLD']:
            save_path = "./{}_model{}_{}.pth".format(net.__class__.__name__, k, TIME)
        else:
            save_path = "./{}_model_{}.pth".format(net.__class__.__name__, TIME)

        fit_model(computing_device, net, criterion, optimizer, train_loader, validation_loader,
                  save_path=save_path)
        nnets.append(net)

    # TODO: Compare models and return the best
    return nnets[0]


def fit_model(computing_device, net, criterion, optimizer, train_loader, validation_loader,
              save_path="./model.pth."):
    """
    Fit the model over a given number of epochs, while saving all loss data in the model itself.
    :param computing_device:
    :param net: nn.Module
    :param criterion: torch.nn
    :param optimizer: torch.optim
    :param train_loader: DataLoader
    :param validation_loader: DataLoader
    :param save_path: string
    :return:
    """
    net.train()
    for epoch in range(settings['EPOCHS']):
        N_minibatch_loss = 0.0
        N = 50

        train_batch_losses = []
        val_batch_losses = []
        avg_minibatch_loss = []
        logging.info("Started new epoch")
        # Get the next minibatch of images, labels for training
        for minibatch_count, (images, labels) in enumerate(train_loader, 0):
            fraction_done = round(minibatch_count / len(train_loader) * 100, 3)
            # Zero out the stored gradient (buffer) from the previous iteration
            optimizer.zero_grad()
            # Put the minibatch data in CUDA Tensors and run on the GPU if supported
            images, labels = images.to(computing_device), labels.to(computing_device)
            # Perform the forward pass through the network and compute the loss
            outputs = net(images)
            loss = criterion(outputs, labels)  # If we are using Cross Entropy, this is doing Softmax
            # Automagically compute the gradients and backpropagate the loss through the network
            loss.backward()

            # Update the weights
            optimizer.step()
            # Add this iteration's loss to the total_loss
            train_batch_losses.append(loss.item())
            N_minibatch_loss += loss

            if minibatch_count % N == 49:
                # Print the loss averaged over the last N mini-batches
                N_minibatch_loss /= N
                logging.info(
                    'Epoch %d, average minibatch %d loss: %.3f' % (epoch + 1, minibatch_count + 1, N_minibatch_loss))
                # Add the averaged loss over N minibatches and reset the counter
                avg_minibatch_loss.append(N_minibatch_loss)
                N_minibatch_loss = 0.0

        logging.info("Finished" + str(epoch + 1) + "epochs of training")
        logging.info("Saving model...")
        torch.save(net.state_dict(), save_path)
        logging.info("Done.")

        # Save this epochs training loss
        train_epoch_loss = np.average(np.array(train_batch_losses))
        train_epoch_losses.append(train_epoch_loss)

        # Validate
        with torch.no_grad():
            net.eval()
            for images, labels in validation_loader:
                # Put the validation mini-batch data in CUDA Tensors and run on the GPU if supported
                images, labels = images.to(computing_device), labels.to(computing_device)
                # Perform the forward pass through the network and compute the loss
                outputs = net(images)

                validation_batch_loss = criterion(outputs, labels)
                val_batch_losses.append(validation_batch_loss.item())

            # Save this epochs validation loss
            val_epoch_loss = np.average(np.array(val_batch_losses))
            val_epoch_losses.append(val_epoch_loss)

        logging.info(
            'Epoch %d Training loss: %.3f Validation loss: %.3f' % (epoch + 1, train_epoch_loss, val_epoch_loss))


def plot_loss(net):
    """
    Plot training- and validation loss over epochs
    :param net: nn.Module
    :return:
    """
    plt.plot(train_epoch_losses, label="train loss")
    plt.plot(val_epoch_losses, label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Loss as a function of number of epochs")
    plt.legend()
    net_name = "TransferNet" if settings['NNET'] is None else settings['NNET'].__name__
    plt.savefig('validation_plot_%s_%s_.png' % (net_name, get_current_time()))


def test(net, test_dataset):
    """
    Test the model on test data, and print statistics
    :param net: nn.Module
    :param test_dataset: torch.utils.data.Dataset
    :return:
    """
    logging.info("Started predicting testing data...")
    computing_device, extra = check_cuda()
    test_loader = DataLoader(test_dataset, batch_size=settings["BATCH_SIZE"], shuffle=False)
    with torch.no_grad():
        all_predictions = []
        all_labels = []
        for images, labels in test_loader:  # Remember they come in batches
            images, labels = images.to(computing_device), labels.to(computing_device)
            # Since we are not doing this through criterion, we must add softmax our self
            outputs = func.softmax(net(images), dim=1)
            _, predicted = torch.max(outputs.data, 1)

            predicted = func.one_hot(predicted, num_classes=settings['NUM_CLASSES']).type(torch.FloatTensor)
            labels = func.one_hot(labels, num_classes=settings['NUM_CLASSES']).type(torch.FloatTensor)
            all_predictions.append(predicted)
            all_labels.append(labels)

        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        logging.info("Evaluating test results...")
        evaluate(all_predictions, all_labels, net, settings)
        sklearn_acc_per_class(all_labels, all_predictions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", "-s", help="If running on server", default=False)
    parser.add_argument("--epochs", "-e", help="Number of epochs", type=int)
    parser.add_argument("--mini", "-m", help="Do you want to run with only 10 classes?", default=False)
    parser.add_argument("--net", "-n", help="AlexNet | Nnet | TransferNet", default="AlexNet")
    parser.add_argument("--kfold", "-k", help="Enable K-fold cross validation", default=False)
    parser.add_argument("--lr", "-lr", help="Learning Rate", type=float)
    parser.add_argument("--decay", "-d", help="Weight Decay", type=float, default=0)
    parser.add_argument("--wloss", "-wl", help="Use weighted loss", default=True)
    parser.add_argument("--transformer", "-t", help="What transformer to run on images", default="default")
    parser.add_argument("--batch", "-b", help="Batch Size", type=int)

    args = parser.parse_args()
    if args.server == "True":
        settings['DATA_PATHS']['DATASET_PATH'] = '/datasets/cs154-fa19-public/'
    if args.net:
        settings['NNET'] = NETS[args.net]
    settings['K-FOLD'] = args.kfold == "True"
    if args.epochs:
        settings['EPOCHS'] = args.epochs
    if args.mini == "True":
        settings['NUM_CLASSES'] = 11
        settings['DATA_PATHS']['TRAIN_CSV'] = "mini_train.csv"
        settings['DATA_PATHS']['TEST_CSV'] = "mini_test.csv"
    if args.lr:
        settings["LR"] = args.lr
    if args.decay:
        settings["DECAY"] = args.decay
    settings["WLOSS"] = args.wloss == "True"
    settings["TRANSFORMER"] = args.transformer
    if args.batch:
        settings["BATCH_SIZE"] = args.batch


    TIME = get_current_time()
    # Load and transform data
    transform = get_transformers()[settings["TRANSFORMER"]]
    dataset = Loader(settings['DATA_PATHS']['TRAIN_CSV'], settings['DATA_PATHS']['DATASET_PATH'], transform=transform)
    test_dataset = Loader(settings['DATA_PATHS']['TEST_CSV'], settings['DATA_PATHS']['DATASET_PATH'],
                          transform=transform)

    # Train k models and keep the best
    logging.info("Settings: {}".format(str(settings)))
    best_model = train(dataset)
    plot_loss(best_model)
    test(best_model, test_dataset)


