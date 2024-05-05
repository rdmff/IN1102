import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

def plot_roc_curves(y_true, y_preds, labels):
    plt.figure(figsize=(8, 6))
    for i in range(len(y_preds)):
        fpr, tpr, _ = roc_curve(y_true, y_preds[i])
        auc_score = roc_auc_score(y_true, y_preds[i])
        plt.plot(fpr, tpr, label=f'{labels[i]} (AUC = {auc_score:.2f})')

    plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

# Dados de teste
y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 0])
# Probabilidades previstas pelos modelos
model1_preds = np.array([0.1, 0.8, 0.6, 0.3, 0.9, 0.2, 0.7, 0.4, 0.85, 0.25])
model2_preds = np.array([0.2, 0.7, 0.55, 0.4, 0.85, 0.3, 0.65, 0.45, 0.8, 0.35])

# Plotando as curvas ROC
plot_roc_curves(y_true, [model1_preds, model2_preds], ['Modelo 1', 'Modelo 2'])
