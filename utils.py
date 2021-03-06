import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
from PIL import Image
from sklearn.model_selection import KFold
from torch.utils.data import Dataset
from torchvision import transforms
import csv
from sklearn.metrics import classification_report, confusion_matrix
import torch
import logging



class Loader(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.transform = transform
        self.frame = pd.read_csv(csv_file)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.frame.iloc[idx, 0])
        img = Image.open(img_name)
        label = self.frame.iloc[idx, 1]

        if self.transform:
            img = self.transform(img)

        return img, label

    def get_class_weights(self):
        """
        Calculate the weight for each class based on number of samples of that class. Also weight = 1
        for the unknown class 0.
        :return: torch.Tensor - size = [#classes + 1,]
        """
        label_counts = self.frame.label.value_counts().sort_index()  # Count images pr label
        class_weights = 1 / torch.Tensor(label_counts.to_list())
        class_weights = torch.cat((torch.Tensor([1.0]), class_weights), 0)
        class_weights = class_weights.float()
        return class_weights

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


def plot_loss(net, settings):
    """
    Plot training- and validation loss over epochs
    :param settings: Settings
    :param net: nn.Module
    :return:
    """
    plt.plot(net.train_epoch_losses, label="train loss")
    plt.plot(net.val_epoch_losses, label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Loss as a function of number of epochs")
    plt.legend()
    net_name = "TransferNet" if settings['NNET'] is None else settings['NNET'].__name__
    plt.savefig('validation_plot_%s_%s_.png' % (net_name, get_current_time()))


def get_transformers():
    max_angle_to_rotate = 30
    dataset_means = [0.44853166, 0.36957467, 0.3424105]
    dataset_std = [0.3184785, 0.28314784, 0.27949163]
    transformers = {"alon": transforms.Compose([transforms.Resize(224),
                                                transforms.CenterCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ColorJitter(),
                                                transforms.RandomRotation(max_angle_to_rotate),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=dataset_means, std=dataset_std)]),
                    "jon": transforms.Compose([
                        transforms.Resize(224),
                        transforms.ColorJitter(),
                        transforms.RandomCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.Resize(128),
                        transforms.ToTensor()
                    ]),
                    "default": transforms.Compose([transforms.Resize(224),
                                                   transforms.CenterCrop(224),
                                                   transforms.ToTensor()])
                    }
    return transformers


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


def get_current_time():
    return datetime.now().strftime("%m.%d.%Y %H:%M:%S")


def get_k_fold_indecies(dataset, random_seed, k=3):
    """
    Return k train-validation splits
    :param random_seed: seed
    :param dataset: Dataset
    :param k: int
    :return:
    """
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    kf = KFold(n_splits=k, shuffle=True, random_state=random_seed)
    return kf.split(indices)


def weights_init(m, xavier=True):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if xavier:
            nn.init.xavier_uniform_(m.weight.data)
        else:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def show_random_images(dataloader):
    """
    Show a grid image with batch_size number of samples.
    :param dataloader: torch.utils.data.DataLoader
    :return:
    """
    dataiter = iter(dataloader)
    images, labels = dataiter.next()  # We get batch_size number of images by calling .next()
    img = torchvision.utils.make_grid(images)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig("image.jpeg")
    plt.show()


def get_accuracy(output, labels):
    """
    The fraction of predictions that are correct on a class basis
    :param output: Predictions from the network - np.ndarray
    :param labels: Labels for the input to the network - np.ndarray
    :return: List[float]
    """
    if not type(output) == np.ndarray: output = np.array(output)
    if not type(labels) == np.ndarray: output = np.array(labels)
    accuracy_per_class = []
    for c in range(output.shape[1]):
        correct = output[:, c] == labels[:, c]
        accuracy_per_class.append(sum(correct) / output.shape[0])
    return accuracy_per_class


def get_precision(output, labels):
    """
    Return the precision score of the predictions per class
    :param output: Predictions from the network
    :param labels: Labels for the input to the network
    :return: List[float]
    """
    truth_per_class = get_truth(output, labels)
    precision_pr_class = []
    for c in range(output.shape[1]):
        TP, FP, TN, FN = truth_per_class[c]
        precision_pr_class.append(TP / (FP + TP + 10 ** (-8)))
    return precision_pr_class


def get_recall(output, labels):
    """
    The recall score of the predictions
    :param output: Predictions from the network.
    :param labels: Labels for the input to the network
    :return: List[float]
    """
    truth_per_class = get_truth(output, labels)
    recall_pr_class = []
    for c in range(output.shape[1]):
        TP, FP, TN, FN = truth_per_class[c]
        recall_pr_class.append(TP / (TP + FN + 10 ** (-8)))
    return recall_pr_class


def get_bcr(output, labels):
    """
    The Balanced Classification Rate per class.
    :param output: Predictions from the network
    :param labels: Labels for the input to the network
    :return: List[float]
    """
    bcrs = []
    precision_pr_class = get_precision(output, labels)
    recal_pr_class = get_recall(output, labels)
    for c in range(output.shape[1]):
        bcr = 0.5 * (precision_pr_class[c] + recal_pr_class[c])
        bcrs.append(bcr)
    return bcrs


def get_truth(output, labels):
    """
    The number of True Positives, False Positives, True Negatives and False Negatives respectfully per class
    :param output: Predictions from the network
    :param labels: Labels for the input to the network
    :return: List[(TP,FP,TN,FN)]
    """
    truth_pr_class = []
    for c in range(output.shape[1]):
        out = output[:, c]
        label = labels[:, c]
        TP = sum(np.logical_and(out, label))
        FP = sum(np.logical_and(out, np.logical_not(label)))
        TN = sum(np.logical_and(np.logical_not(out), np.logical_not(label)))
        FN = sum(np.logical_and(np.logical_not(out), label))
        truth_pr_class.append((TP, FP, TN, FN))
    return truth_pr_class


def get_aggregated_score(output, labels):
    """
    The aggregated score for precision, recall, and BCR by computing an equally weighted average across per class
    all individual scores, respectfully.
    :param output: Predictions from the network
    :param labels: Labels for the input to the network
    :return: List[float]
    """
    accuracy = get_accuracy(output, labels)
    precision = get_precision(output, labels)
    recall = get_recall(output, labels)
    bcr = get_bcr(output, labels)

    aggregated_scores = []
    for c in range(output.shape[1]):
        score = (accuracy[c] + precision[c] + recall[c] + bcr[c]) / 4
        aggregated_scores.append(score)
    return aggregated_scores


def evaluate(output, labels, net, settings):
    """
    Print the overall score of the predictions done by the network
    :param output: Predictions from the network
    :param labels: Labels for the input to the network
    :return:
    """
    headers = ["Class", "Accuracy", "Precision", "Recall", "BCR", "AggregatedScore"]
    # True positives, false positives, etc.
    truth_per_class = get_truth(output, labels)
    evaluate_per_class = []
    acc_per_class = sklearn_acc_per_class(labels, output)
    for c in range(output.shape[1]):
        TP, FP, TN, FN = list(map(lambda x: x.item(), truth_per_class[c]))
        precision = TP / (FP + TP + 10 ** (-8))
        recall = TP / (TP + FN + 10 ** (-8))
        bcr = 0.5 * (precision + recall)
        aggregated_score = (acc_per_class[c] + precision + recall + bcr) / 4
        evaluate_per_class.append((c, acc_per_class[c], precision, recall, bcr, aggregated_score))

    # Save to file as well as print to console
    result_file = open("{}_test_results_{}.csv".format(net.__class__.__name__, get_current_time()), mode="w")
    csv_writer = csv.writer(result_file, delimiter=',')
    csv_writer.writerow([str(settings)])
    csv_writer.writerow(headers)
    dash = "-" * 65
    print(dash)
    print('{:<10s}{:<10s}{:<10s}{:<10s}{:<10s}{:<10s}'.format(*headers))
    print(dash)
    for class_eval in evaluate_per_class:
        csv_writer.writerow([round(num, ndigits=3) for num in class_eval])
        print('{:<10d}{:<10.3f}{:<10.3f}{:<10.3f}{:<10.3f}{:<10.3f}'.format(*class_eval))
    result_file.close()


def sklearn_metrics(y_true, y_pred):
    print(classification_report(y_true, y_pred))


def sklearn_acc_per_class(y_true, y_pred):
    _, y_pred = y_pred.max(1)
    _, y_true = y_true.max(1)
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    acc_per_class = cm.diagonal()
    acc_per_class = np.concatenate((np.array([0]), acc_per_class), axis=None)
    return acc_per_class


def plot_weights(model, layer):
    COLUMNS_IN_FIGURE = 10
    weights = model._modules['main']._modules[str(layer)].weight.data.numpy()

    # # normalize weights
    # mean = np.mean(weights, axis=(2, 3), keepdims=True)
    # std = np.std(weights, axis=(2, 3), keepdims=True)
    # weights = (weights - mean) / std

    num_weights = weights.shape[0]
    num_rows = 1 + num_weights // COLUMNS_IN_FIGURE
    fig = plt.figure(figsize=(COLUMNS_IN_FIGURE, num_rows))
    for i in range(weights.shape[1]):
        sub = fig.add_subplot(num_rows, COLUMNS_IN_FIGURE, i + 1)
        sub.axis('off')
        sub.imshow(weights[0][i], cmap='gray')
        sub.set_xticklabels([])
        sub.set_yticklabels([])

    plt.show()


def main():
    pred = np.eye(20)[np.random.choice(20, 1000)]
    labels = np.eye(20)[np.random.choice(20, 1000)]
    evaluate(pred, labels)
