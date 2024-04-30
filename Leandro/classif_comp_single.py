{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\"\"\"\n",
    "ALGORITMOS PARA CÁLCULO DE COMPARAÇÃO ENTRE DOIS CLASSIFICADORES\n",
    "Desenvolvido pelos DrC: Roberto de Medeiros F F; Ana Claudia S L Santos; Francisco A S Rodrigues; Rondynelly Duarte O J.\n",
    "Turma: IN1102 2024.1 CIn UFPE\n",
    "Prof.: Leandro Maciel Almeida, PhD\n",
    "\"\"\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "from scipy.stats import f_oneway\n",
    "from statsmodels.stats.multicomp import pairwise_tukeyhsd\n",
    "\n",
    "def ANOVA_two(score_c1,score_c2):#Tupla (Acuracy, Precision, Recall, F1-score, AUC ROC, ...)\n",
    "    # Dados de desempenho dos classificadores (por exemplo, precisão)\n",
    "    #classificador1 = [82, 85, 88, 90, 86]\n",
    "    #classificador2 = [80, 84, 87, 88, 85]\n",
    "\n",
    "    # Teste ANOVA\n",
    "    f_statistic, p_value = f_oneway(score_c1, score_c2)\n",
    "\n",
    "    # Interpretar resultados do ANOVA\n",
    "    if p_value < 0.05:\n",
    "        print(\"Há diferença estatisticamente significativa entre os classificadores.\")\n",
    "    else:\n",
    "        print(\"Não há diferença estatisticamente significativa entre os classificadores.\")\n",
    "\n",
    "    \"\"\"\n",
    "    [C2C1] OLIVEIRA, Bruno. Teste de Tukey para Comparações Múltiplas. Site statplace. Publicado 21 de ago. de 2019.\n",
    "        Disponível em https://statplace.com.br/blog/comparacoes-multiplas-teste-de-tukey/. Acesso em 30 abr. de 2024.\n",
    "        \n",
    "    'O Teste de Tukey consiste em comparar todos os possíveis pares de médias e se baseia na\n",
    "        diferença mínima significativa (D.M.S.), considerando os percentis do grupo.\n",
    "        No cálculo da D.M.S. utiliza-se também a distribuição da amplitude estudentizada,\n",
    "        o quadrado médio dos resíduos da ANOVA e o tamanho amostral dos grupos.' (C2C1)\n",
    "    DMS = Diferença Mínima Significativa\n",
    "    \"\"\"\n",
    "    # Executar teste de Tukey para testes post-hoc\n",
    "    data = np.array(classificador1 + classificador2)\n",
    "    labels = ['Classificador 1'] * len(classificador1) + ['Classificador 2'] * len(classificador2)\n",
    "    tukey_results = pairwise_tukeyhsd(data, labels, 0.05)\n",
    "\n",
    "    return \"\\n Tukey\" + tukey_results"
   ]
  }
 ],
 "metadata": {
  "colab": {
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
