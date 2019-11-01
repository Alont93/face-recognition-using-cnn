import numpy as np


def get_accuracy(output, labels):
    """
    The fraction of predictions that are correct
    :param output: Predictions from the network - np.ndarray
    :param labels: Labels for the input to the network - np.ndarray
    :return: float
    """
    if not type(output) ==  np.ndarray: output = np.array(output)
    if not type(labels) == np.ndarray: output = np.array(labels)
    correct = output == labels
    return sum(correct) / len(output)


def get_precision(output, labels):
    """
    Return the precision score of the predictions
    :param output: Predictions from the network
    :param labels: Labels for the input to the network
    :return: float
    """
    TP, FP, TN, FN = get_truth(output, labels)
    return TP / (FP + TP)


def get_recall(output, labels):
    """
    The recall score of the predictions
    :param output: Predictions from the network.
    :param labels: Labels for the input to the network
    :return: float
    """
    TP, FP, TN, FN = get_truth(output, labels)
    return TP / (TP + FN)


def get_bcr(output, labels):
    """
    The Balanced Classification Rate.
    :param output: Predictions from the network
    :param labels: Labels for the input to the network
    :return: float
    """
    return 0.5 * (get_precision(output, labels) + get_recall(output, labels))


def get_truth(output, labels):
    """
    The number of True Positives, False Positives, True Negatives and False Negatives respectfully.
    :param output: Predictions from the network
    :param labels: Labels for the input to the network
    :return: floats
    """
    TP = np.logical_and(output, labels)
    FP = np.logical_and(output, np.logical_not(labels))
    TN = np.logical_and(np.logical_not(output), np.logical_not(labels))
    FN = np.logical_and(np.logical_not(output), labels)
    return sum(TP), sum(FP), sum(TN), sum(FN)


def get_aggregated_score(output, labels):
    """
    The aggregated score for precision, recall, and BCR by computing an equally weighted average across
    all individual scores, respectfully.
    :param output: Predictions from the network
    :param labels: Labels for the input to the network
    :return: floats
    """
    accuracy = get_accuracy(output, labels)
    precision = get_precision(output, labels)
    recall = get_recall(output, labels)
    bcr = get_bcr(output, labels)
    return (accuracy + precision + recall + bcr) / 4


def evaluate(output, labels):
    """
    Print the overall score of the predictions done by the network
    :param output: Predictions from the network
    :param labels: Labels for the input to the network
    :return:
    """
    # True positives, false positives, etc.
    TP, FP, TN, FN = get_truth(output, labels)

    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (FP + TP)
    recall = TP / (TP + FN)
    bcr = 0.5 * (precision + recall)
    aggregated_score = (accuracy + precision + recall + bcr) / 4
    dash = "-" * 55
    print(dash)
    print('{:<10s}{:<10s}{:<10s}{:<10s}{:<10s}'.format("Accuracy", "Precision", "Recall", "BCR", "AggregatedScore"))
    print(dash)
    print('{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}'.format(accuracy, precision, recall, bcr, aggregated_score))


def main():
    pred = np.array([1, 1, 1, 1])
    label = np.array([1, 1, 1, 0])
    evaluate(pred, label)

# main()
