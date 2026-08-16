"""
evaluate.py
-----------
 
funtion in thid file :
    - calculate_accuracy   : Calculate the model accuracy
    - plot_k_curve          : Plot a graph comparing the accuracy for each value of k.
    - plot_confusion_matrix : Plot the confusion matrix of the best-performing model.
"""
 
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
 
 
def calculate_accuracy(y_true, y_pred):
    """
    calculate accuracy = Number of correct predictions / Total number of samples
    """
    return accuracy_score(y_true, y_pred)
 
 
def plot_k_curve(k_values, accuracies, save_path):
    """
    paramiter:
        k_values (list): K value = [3, 5, 7]
        accuracies (list): accuracy for K value
        save_path (str): position save file
    """
    plt.figure(figsize=(7, 5))
    plt.plot(k_values, accuracies, marker='o')
    plt.title('Accuracy vs k value')
    plt.xlabel('k (number of neighbors)')
    plt.ylabel('Accuracy')
    plt.xticks(k_values)
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
 
 
def plot_confusion_matrix(y_true, y_pred, save_path):
    """
    paramiter:
        y_true (ndarray): ture answer
        y_pred (ndarray): prediction anser
        save_path (str): position save file
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix (Best k)')
    plt.savefig(save_path)
    plt.close()
 