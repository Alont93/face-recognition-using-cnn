import argparse
import logging
from baseline_cnn import *
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as torch_init
import torch.optim as optim

# Data utils and dataloader
from torchvision import transforms

# Custom utils file
from utils import evaluate, weights_init

logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

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


def rand_train_val_split(dataset, validation_split=0.2, shuffle_dataset=True, random_seed=42):
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    return train_indices, val_indices


def train(dataset, weighted_loss=False, epochs=10, batch_size=64):
    computing_device, extra = check_cuda()

    # TODO: Init k set of indecies instead of one
    train_indecies, val_indecies = rand_train_val_split(dataset)
    k_train_indecies = [train_indecies]
    k_val_indecies = [val_indecies]
    # TODO: For K in k-fold (Make lists of all the indecies before this loop)
    nnets = []
    for k, (train_indices, val_indices) in enumerate(zip(k_train_indecies, k_val_indecies)):
        logging.info("Training Model {}".format(k))
        # Load data for this fold
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

        # Initialize CNN
        net = Nnet(num_classes=NUM_CLASSES).to(computing_device)
        net.apply(weights_init)

        # Initialize optimizer and criterion
        if weighted_loss:
            criterion = criterion = nn.CrossEntropyLoss(weight=dataset.get_class_weights())
        else:
            criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=0.0005)

        save_path = "./model{}.pth".format(k)
        fit_model(computing_device, net, criterion, optimizer, train_loader, validation_loader, save_path=save_path)
        nnets.append(net)

    # TODO: Return the best model
    return nnets[0]


def fit_model(computing_device, net, criterion, optimizer, train_loader, validation_loader, epochs=50,
              save_path="./model.pth."):
    """
    Fit the model over a given number of epochs, while saving all loss data in the model itself.
    :param computing_device:
    :param net: nn.Module
    :param criterion: torch.nn
    :param optimizer: torch.optim
    :param train_loader: torch.utils.data.DataLoader
    :param validation_loader: torch.utils.data.DataLoader
    :param epochs: int
    :param save_path: string
    :return:
    """
    # Initalize k models

    for epoch in range(epochs):

        N_minibatch_loss = 0.0
        N = 50

        train_batch_losses = []
        val_batch_losses = []
        avg_minibatch_loss = []
        # Get the next minibatch of images, labels for training
        for minibatch_count, (images, labels) in enumerate(train_loader, 0):
            fraction_done = round(minibatch_count / len(train_loader) * 100, 3)
            print("{} percent of epoch {} complete".format(fraction_done, epoch + 1), end="\r")
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
                logging.info('Epoch %d, average minibatch %d loss: %.3f' % (epoch + 1, minibatch_count + 1, N_minibatch_loss))
                # Add the averaged loss over N minibatches and reset the counter
                avg_minibatch_loss.append(N_minibatch_loss)
                N_minibatch_loss = 0.0

        logging.info("Finished", epoch + 1, "epochs of training")
        logging.info("Saving model...")
        torch.save(net.state_dict(), save_path)
        logging.info("Done.")

        # Save this epochs training loss
        train_epoch_loss = np.average(np.array(train_batch_losses))
        net.train_epoch_losses.append(train_epoch_loss)

        # Validate
        with torch.no_grad():
            for _, (images, labels) in enumerate(validation_loader, 0):
                # Put the validation mini-batch data in CUDA Tensors and run on the GPU if supported
                images, labels = images.to(computing_device), labels.to(computing_device)
                # Perform the forward pass through the network and compute the loss
                outputs = net(images)

                validation_batch_loss = criterion(outputs, labels)
                val_batch_losses.append(validation_batch_loss)

            # Save this epochs validation loss
            val_epoch_loss = np.average(np.array(val_batch_losses))
            net.validation_epoch_losses.append(val_epoch_loss)

        logging.info('Epoch %d Training loss: %.3f Validation loss: %.3f' % (epoch + 1, train_epoch_loss, val_epoch_loss))


def test(net, test_dataset):
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    with torch.no_grad():
        for images, labels in test_loader:  # Remember they come in batches
            # Since we are not doing this through criterion, we must add softmax our self
            outputs = func.softmax(net(images), dim=1)
            _, predicted = torch.max(outputs.data, 1)

            predicted = func.one_hot(predicted, num_classes=NUM_CLASSES).type(torch.FloatTensor)
            labels = func.one_hot(labels, num_classes=NUM_CLASSES).type(torch.FloatTensor)
            evaluate(predicted, labels)


if __name__ == '__main__':
    EPOCHS = 1
    BATCH_SIZE = 64
    NUM_CLASSES = 11
    LOCAL = True

    parser = argparse.ArgumentParser()
    parser.add_argument("--local", "-l", help="If running locally or on ieng6")
    args = parser.parse_args()
    if args.local:
        LOCAL = bool(args.local)

    transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor()])

    dataset_path = './datasets/cs154-fa19-public/' if LOCAL else '/datasets/cs154-fa19-public/'


    dataset = Loader('mini_train.csv', dataset_path, transform=transform)
    test_dataset = Loader("mini_test.csv", dataset_path, transform=transform)

    # Train k models and keep the best
    best_model = train(dataset)
