Abaixo, seguem exemplos de uso das métricas principais (acurácia já usávamos).

Precisão, Recall e F1-score:
Precisão, recall e F1-score são métricas comumente usadas para avaliar modelos de classificação.


from sklearn.metrics import precision_score, recall_score, f1_score

# Exemplo de predições e rótulos verdadeiros
y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]

# Calcular precisão
precision = precision_score(y_true, y_pred)
print("Precisão:", precision)

# Calcular recall
recall = recall_score(y_true, y_pred)
print("Recall:", recall)

# Calcular F1-score
f1 = f1_score(y_true, y_pred)
print("F1-score:", f1)
AUC-ROC (Área sob a curva ROC):
A AUC-ROC é uma métrica que avalia o desempenho de um classificador binário em vários limiares de discriminação.





from sklearn.metrics import roc_auc_score

# Exemplo de pontuações de probabilidade previstas e rótulos verdadeiros
y_true = [0, 1, 1, 0, 1]
y_score = [0.1, 0.9, 0.8, 0.3, 0.7]  # Probabilidade de classe positiva

# Calcular AUC-ROC
auc_roc = roc_auc_score(y_true, y_score)
print("AUC-ROC:", auc_roc)
