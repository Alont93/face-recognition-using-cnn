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
import numpy as np

# Custom files
from models import Nnet, TronNet
from utils import evaluate, weights_init, get_k_fold_indecies, get_transformers, get_current_time, \
    sklearn_acc_per_class, check_cuda, plot_loss, Loader

TIME = None

NETS = {
    "Nnet": Nnet,
    "TransferNet": None,
    "TronNet": TronNet
}

SETTINGS = {
    'EPOCHS': 50,
    'BATCH_SIZE': 64,
    'LR': 0.001,
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
    for epoch in range(SETTINGS['EPOCHS']):
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
        net.train_epoch_losses.append(train_epoch_loss)

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
            net.val_epoch_losses.append(val_epoch_loss)

        logging.info(
            'Epoch %d Training loss: %.3f Validation loss: %.3f' % (epoch + 1, train_epoch_loss, val_epoch_loss))


def train(dataset):
    computing_device, extra = check_cuda()

    # Save all the k models to compare
    nnets = []
    batch_size = SETTINGS['BATCH_SIZE']

    # Get a lists of train-val-split for k folds
    if SETTINGS['K-FOLD']:
        indices = get_k_fold_indecies(dataset, SETTINGS['RANDOM_SEED'], k=SETTINGS['K-FOLD-NUMBER'])
    else:
        indices = list(range(len(dataset)))
        validation_split = .1
        split = int(np.floor(validation_split * len(dataset)))
        np.random.seed(SETTINGS['RANDOM_SEED'])
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
        if SETTINGS['NNET'] is None:
            net = models.resnet152(pretrained=True)

            # Freeze parameters, so gradient not computed here
            for param in net.parameters():
                param.requires_grad = False

            net.fc = nn.Linear(net.fc.in_features, SETTINGS['NUM_CLASSES'])
            net = net.to(computing_device)
            net.train_epoch_losses = []
            net.val_epoch_losses = []
        else:
            net = SETTINGS['NNET'](SETTINGS['NUM_CLASSES']).to(computing_device)
            net.apply(weights_init)

        # Initialize optimizer and criterion
        if SETTINGS["WLOSS"]:
            criterion = nn.CrossEntropyLoss(weight=dataset.get_class_weights().to(computing_device))
        else:
            criterion = nn.CrossEntropyLoss()

        parameters_to_learn = []
        if SETTINGS['NNET'] is None:
            for name, param in net.named_parameters():
                if param.requires_grad:
                    parameters_to_learn.append(param)
        else:
            parameters_to_learn = net.parameters()

        optimizer = optim.Adam(parameters_to_learn, lr=SETTINGS["LR"], weight_decay=SETTINGS["DECAY"])


        # Fit and save model to file
        if SETTINGS['K-FOLD']:
            save_path = "./{}_model{}_{}.pth".format(net.__class__.__name__, k, TIME)
        else:
            save_path = "./{}_model_{}.pth".format(net.__class__.__name__, TIME)

        fit_model(computing_device, net, criterion, optimizer, train_loader, validation_loader,
                  save_path=save_path)
        nnets.append(net)

    best_net = None
    for nnet in nnets:
        if best_net is None or min(nnet.val_epoch_losses) < min(best_net.val_epoch_losses):
            best_net = nnet
    return best_net


def test(net, test_dataset):
    """
    Test the model on test data, and print statistics
    :param net: nn.Module
    :param test_dataset: torch.utils.data.Dataset
    :return:
    """
    logging.info("Started predicting testing data...")
    computing_device, extra = check_cuda()
    test_loader = DataLoader(test_dataset, batch_size=SETTINGS["BATCH_SIZE"], shuffle=False)
    with torch.no_grad():
        net.eval()
        all_predictions = []
        all_labels = []
        for images, labels in test_loader:  # Remember they come in batches
            images, labels = images.to(computing_device), labels.to(computing_device)

            # Since we are not doing this through criterion, we must add softmax our self
            outputs = func.softmax(net(images), dim=1)
            _, predicted = torch.max(outputs.data, 1)

            predicted = func.one_hot(predicted, num_classes=SETTINGS['NUM_CLASSES']).type(torch.FloatTensor)
            labels = func.one_hot(labels, num_classes=SETTINGS['NUM_CLASSES']).type(torch.FloatTensor)
            all_predictions.append(predicted)
            all_labels.append(labels)

        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        logging.info("Evaluating test results...")
        evaluate(all_predictions, all_labels, net, SETTINGS)
        sklearn_acc_per_class(all_labels, all_predictions)


if __name__ == '__main__':
    logging.basicConfig(filename='app.log', filemode='w', format='%(asctime)s %(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    parser = argparse.ArgumentParser()
    parser.add_argument("--server", "-s", help="If running on server", default=False)
    parser.add_argument("--epochs", "-e", help="Number of epochs", type=int)
    parser.add_argument("--mini", "-m", help="Do you want to run with only 10 classes?", default=False)
    parser.add_argument("--net", "-n", help="TronNet | Nnet | TransferNet", default="TronNet")
    parser.add_argument("--kfold", "-k", help="Enable K-fold cross validation", default=False)
    parser.add_argument("--lr", "-lr", help="Learning Rate", type=float)
    parser.add_argument("--decay", "-d", help="Weight Decay", type=float, default=0)
    parser.add_argument("--wloss", "-wl", help="Use weighted loss", default=True)
    parser.add_argument("--transformer", "-t", help="What transformer to run on images", default="default")
    parser.add_argument("--batch", "-b", help="Batch Size", type=int)

    args = parser.parse_args()
    if args.server == "True":
        SETTINGS['DATA_PATHS']['DATASET_PATH'] = '/datasets/cs154-fa19-public/'
    if args.net:
        SETTINGS['NNET'] = NETS[args.net]
    SETTINGS['K-FOLD'] = args.kfold == "True"
    if args.epochs:
        SETTINGS['EPOCHS'] = args.epochs
    if args.mini == "True":
        SETTINGS['NUM_CLASSES'] = 11
        SETTINGS['DATA_PATHS']['TRAIN_CSV'] = "mini_train.csv"
        SETTINGS['DATA_PATHS']['TEST_CSV'] = "mini_test.csv"
    if args.lr:
        SETTINGS["LR"] = args.lr
    if args.decay:
        SETTINGS["DECAY"] = args.decay
    SETTINGS["WLOSS"] = args.wloss == "True"
    SETTINGS["TRANSFORMER"] = args.transformer
    if args.batch:
        SETTINGS["BATCH_SIZE"] = args.batch

    TIME = get_current_time()

    # Load and transform data
    transform = get_transformers()[SETTINGS["TRANSFORMER"]]
    dataset = Loader(SETTINGS['DATA_PATHS']['TRAIN_CSV'], SETTINGS['DATA_PATHS']['DATASET_PATH'], transform=transform)
    test_dataset = Loader(SETTINGS['DATA_PATHS']['TEST_CSV'], SETTINGS['DATA_PATHS']['DATASET_PATH'],
                          transform=transform)

    # Train k models and keep the best
    logging.info("Settings: {}".format(str(SETTINGS)))
    best_model = train(dataset)
    plot_loss(best_model, SETTINGS)
    test(best_model, test_dataset)
