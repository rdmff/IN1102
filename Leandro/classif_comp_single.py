""" 
ALGORITMOS PARA CÁLCULO DE COMPARAÇÃO ENTRE DOIS CLASSIFICADORES
Desenvolvido pelos DrC: Roberto de Medeiros F F; Ana Claudia S L Santos; Francisco A S Rodrigues; Rondinelly Duarte O J.
Turma: IN1102 2024.1 CIn UFPE
Prof.: Leandro Maciel Almeida, PhD
"""

import numpy as np
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def ANOVA_two(score_c1,score_c2):#Tupla (Acuracy, Precision, Recall, F1-score, AUC ROC, ...)
    # Dados de desempenho dos classificadores (por exemplo, precisão)
    #classificador1 = [82, 85, 88, 90, 86]
    #classificador2 = [80, 84, 87, 88, 85]

    # Teste ANOVA\n,
    f_statistic, p_value = f_oneway(score_c1, score_c2)

    # Interpretar resultados do ANOVA
    if p_value < 0.05:
        print("Há diferença estatisticamente significativa entre os classificadores.")
    else:
        print("Não há diferença estatisticamente significativa entre os classificadores.")
    """
    [C2C1] OLIVEIRA, Bruno. Teste de Tukey para Comparações Múltiplas. Site statplace. Publicado 21 de ago. de 2019.
        Disponível em https://statplace.com.br/blog/comparacoes-multiplas-teste-de-tukey/. Acesso em 30 abr. de 2024.
    
    'O Teste de Tukey consiste em comparar todos os possíveis pares de médias e se baseia na
        diferença mínima significativa (D.M.S.), considerando os percentis do grupo.
        No cálculo da D.M.S. utiliza-se também a distribuição da amplitude estudentizada,
        o quadrado médio dos resíduos da ANOVA e o tamanho amostral dos grupos.' (C2C1)
        
    DMS = Diferença Mínima Significativa
    """
    # Executar teste de Tukey para testes post-hoc
    data = np.array(classificador1 + classificador2)
    labels = ['Classificador 1'] * len(classificador1) + ['Classificador 2'] * len(classificador2)
    tukey_results = pairwise_tukeyhsd(data, labels, 0.05)

    return "Tukey" + tukey_results
