"""Evaluation metrics for a binary classifier.

Labels are lists of 0s and 1s, where 1 is the positive class.
"""


def calculate_accuracy(y_true, y_pred):
    """Return the share of predictions that match the true labels."""
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def calculate_precision(y_true, y_pred):
    """Return the share of positive predictions that were correct."""
    true_positives = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    predicted_positives = sum(1 for p in y_pred if p == 1)
    if predicted_positives == 0:
        return 0.0
    return true_positives / predicted_positives


def calculate_recall(y_true, y_pred):
    """Return the share of actual positives the model found."""
    true_positives = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    actual_positives = sum(1 for t in y_true if t == 1)
    if actual_positives == 0:
        return 0.0
    return true_positives / actual_positives


# TODO

def calculate_f1(y_true, y_pred):
    """Added F1 score calculation to evaluation module"""
    precision = calculate_precision(y_true, y_pred)
    recall = calculate_recall(y_true, y_pred)

    if precision == 0 and recall == 0:
        return 0
    else:
        F1 = 2 * (precision * recall) / (precision + recall)
        return F1

