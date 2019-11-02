import numpy as np


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


def evaluate(output, labels):
    """
    Print the overall score of the predictions done by the network
    :param output: Predictions from the network
    :param labels: Labels for the input to the network
    :return:
    """
    # True positives, false positives, etc.
    truth_per_class = get_truth(output, labels)
    evaluate_per_class = []
    for c in range(output.shape[1]):
        TP, FP, TN, FN = truth_per_class[c]
        accuracy = (TP + TN) / (TP + FP + TN + FN)
        precision = TP / (FP + TP + 10 ** (-8))
        recall = TP / (TP + FN + 10 ** (-8))
        bcr = 0.5 * (precision + recall)
        aggregated_score = (accuracy + precision + recall + bcr) / 4
        evaluate_per_class.append((c, accuracy, precision, recall, bcr, aggregated_score))

    dash = "-" * 65
    print(dash)
    print('{:<10s}{:<10s}{:<10s}{:<10s}{:<10s}{:<10s}'.format("Class", "Accuracy", "Precision", "Recall", "BCR",
                                                              "AggregatedScore"))
    print(dash)
    for class_eval in evaluate_per_class:
        print('{:<10d}{:<10.3f}{:<10.3f}{:<10.3f}{:<10.3f}{:<10.3f}'.format(*class_eval))


def main():
    pred = np.eye(20)[np.random.choice(20, 1000)]
    labels = np.eye(20)[np.random.choice(20, 1000)]
    evaluate(pred, labels)


# main()