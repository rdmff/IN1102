Vamos por os ynpb aqui e comparar.

Códigos - Rondinelly
OBS.: Utilizei a base de dados já normalizada seguindo o mesmo código de Ana.

KNN:
########################################################################################################################################################################################################
from sklearn.metrics import classification_report

# we create an instance of Neighbours Classifier and fit the data.
clf = neighbors.KNeighborsClassifier()

# treinando o classificador
clf = clf.fit(X_train, y_train)

# construindo o espaco de busca por configuracoes do classificador
k_range = range(1, 30)
distancias = ["euclidean", "manhattan", "cosine", "minkowski"]
resultados = {}

for distancia in distancias:
    k_scores_train = []
    k_scores_train_full = []
    k_scores_valid = []
    k_scores_test = []

    for k in k_range:
        knn = neighbors.KNeighborsClassifier(n_neighbors=k, metric=distancia)
        scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
        k_scores_train.append(scores.mean())
        knn.fit(X_train, y_train)
        k_scores_train_full.append(knn.score(X_train, y_train))
        k_scores_valid.append(knn.score(X_valid, y_valid))
        k_scores_test.append(knn.score(X_test, y_test))

    resultados[distancia] = {
        "k_range": k_range,
        "k_scores_train": k_scores_train,
        "k_scores_train_full": k_scores_train_full,
        "k_scores_valid": k_scores_valid,
        "k_scores_test": k_scores_test
    }

# Encontrar a melhor configuração para todas as distâncias
melhor_metrica_global = 0
melhor_configuracao = None

for distancia, resultado in resultados.items():
    melhor_k_index = np.argmax(resultado["k_scores_valid"])
    melhor_k = resultado["k_range"][melhor_k_index]
    melhor_accuracy_train = resultado["k_scores_train_full"][melhor_k_index]
    melhor_accuracy_valid = resultado["k_scores_valid"][melhor_k_index]
    melhor_accuracy_test = resultado["k_scores_test"][melhor_k_index]

    # Treinar o modelo final com os melhores hiperparâmetros e calcular as métricas de classificação
    knn_final = neighbors.KNeighborsClassifier(n_neighbors=melhor_k, metric=distancia)
    knn_final.fit(X_train, y_train)
    y_pred_train = knn_final.predict(X_train)
    y_pred_test = knn_final.predict(X_test)
    report_train = classification_report(y_train, y_pred_train, output_dict=True)
    report_test = classification_report(y_test, y_pred_test, output_dict=True)
    f1_score_metrica = report_test['macro avg']['f1-score']

    # Se a métrica F1-score for melhor do que a melhor até agora, atualize a melhor configuração
    if f1_score_metrica > melhor_metrica_global:
        melhor_metrica_global = f1_score_metrica
        melhor_configuracao = {
            "distancia": distancia,
            "melhor_k": melhor_k,
            "accuracy_train": melhor_accuracy_train,
            "accuracy_valid": melhor_accuracy_valid,
            "accuracy_test": melhor_accuracy_test,
            "precision": report_test['macro avg']['precision'],
            "recall": report_test['macro avg']['recall'],
            "f1-score": f1_score_metrica
        }

# Imprimir métricas para o melhor modelo
print(f"Melhor configuração usando {melhor_configuracao['distancia']} com k={melhor_configuracao['melhor_k']}:")
print("Acurácia de treinamento:", melhor_configuracao["accuracy_train"])
print("Acurácia de validação:", melhor_configuracao["accuracy_valid"])
print("Acurácia de teste:", melhor_configuracao["accuracy_test"])
print("Precisão:", melhor_configuracao["precision"])
print("Recall:", melhor_configuracao["recall"])
print("F1-score:", melhor_configuracao["f1-score"])

# Imprimir métricas para todos os modelos
for distancia, resultado in resultados.items():
    print(f"\nMétricas para a distância {distancia} com k={resultado['k_range'][np.argmax(resultado['k_scores_valid'])]}:")
    print("Acurácia de treinamento:", resultado["k_scores_train_full"][np.argmax(resultado['k_scores_valid'])])
    print("Acurácia de validação:", resultado["k_scores_valid"][np.argmax(resultado['k_scores_valid'])])
    print("Acurácia de teste:", resultado["k_scores_test"][np.argmax(resultado['k_scores_valid'])])
    print("Precisão:", report_test['macro avg']['precision'])
    print("Recall:", report_test['macro avg']['recall'])
    print("F1-score:", report_test['macro avg']['f1-score'])

########################################################################################################################################################################################################
Resultados:
Melhor configuração usando manhattan com k=29:
Acurácia de treinamento: 0.28152412280701755
Acurácia de validação: 0.20087719298245615
Acurácia de teste: 0.19271929824561404
Precisão: 0.18414719516631933
Recall: 0.19706747112325274
F1-score: 0.18053756449144476

Métricas para a distância euclidean com k=25:
Acurácia de treinamento: 0.25699013157894735
Acurácia de validação: 0.1712719298245614
Acurácia de teste: 0.1687719298245614
Precisão: 0.16195955203506363
Recall: 0.17224678971789106
F1-score: 0.1590557757275467

Métricas para a distância manhattan com k=29:
Acurácia de treinamento: 0.28152412280701755
Acurácia de validação: 0.20087719298245615
Acurácia de teste: 0.19271929824561404
Precisão: 0.16195955203506363
Recall: 0.17224678971789106
F1-score: 0.1590557757275467

Métricas para a distância cosine com k=13:
Acurácia de treinamento: 0.3002467105263158
Acurácia de validação: 0.17203947368421052
Acurácia de teste: 0.1712280701754386
Precisão: 0.16195955203506363
Recall: 0.17224678971789106
F1-score: 0.1590557757275467

Métricas para a distância minkowski com k=25:
Acurácia de treinamento: 0.25699013157894735
Acurácia de validação: 0.1712719298245614
Acurácia de teste: 0.1687719298245614
Precisão: 0.16195955203506363
Recall: 0.17224678971789106
F1-score: 0.1590557757275467
########################################################################################################################################################################################################

LVQ:
########################################################################################################################################################################################################
###Funções:
# LVQ for the Spotify dataset
from random import seed
from random import randrange
from csv import reader
from math import sqrt

# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset


# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores

#Distancia de Manhattan
def manhattan_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += abs(row1[i] - row2[i])
    return distance

#Distancia de Chebyshev
#def chebyshev_distance(row1, row2):
 #   distance = 0.0
  #  for i in range(len(row1)-1):
   #     distance = max(distance, abs(row1[i] - row2[i]))
    #return distance

## calculate the Euclidean distance between two vectors
#def euclidean_distance(row1, row2):
	#distance = 0.0
	#for i in range(len(row1)-1):
	#	distance += (row1[i] - row2[i])**2
	#return sqrt(distance)

# Locate the best matching unit
def get_best_matching_unit(codebooks, test_row):
	distances = list()
	for codebook in codebooks:
		dist = manhattan_distance(codebook, test_row)
		distances.append((codebook, dist))
	distances.sort(key=lambda tup: tup[1])
	return distances[0][0]

# Make a prediction with codebook vectors
def predict(codebooks, test_row):
	bmu = get_best_matching_unit(codebooks, test_row)
	return bmu[-1]

# Create a random codebook vector
def random_codebook(train):
	n_records = len(train)
	n_features = len(train[0])
	codebook = [train[randrange(n_records)][i] for i in range(n_features)]
	return codebook

# Train a set of codebook vectors
def train_codebooks(train, n_codebooks, lrate, epochs):
	codebooks = [random_codebook(train) for i in range(n_codebooks)]
	for epoch in range(epochs):
		rate = lrate * (1.0-(epoch/float(epochs)))
		for row in train:
			bmu = get_best_matching_unit(codebooks, row)
			for i in range(len(row)-1):
				error = row[i] - bmu[i]
				if bmu[-1] == row[-1]:
					bmu[i] += rate * error
				else:
					bmu[i] -= rate * error
	return codebooks

# LVQ Algorithm
def learning_vector_quantization(train, test, n_codebooks, lrate, epochs):
	codebooks = train_codebooks(train, n_codebooks, lrate, epochs)
	predictions = list()
	for row in test:
		output = predict(codebooks, row)
		predictions.append(output)
	return(predictions)
########################################################################################################################################################################################################
### Segunda etapa:
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# Selecionar apenas as colunas numéricas
data_numerico = df.select_dtypes(include=['float64', 'int64'])
#
## Padronização
scaler = StandardScaler()
data_padronizado = scaler.fit_transform(data_numerico)
print("Dados padronizados:")
print(data_padronizado)

# Normalização
scaler = MinMaxScaler()
data_normalizado = scaler.fit_transform(data_numerico)
print("\nDados normalizados:")
print(data_normalizado)
########################################################################################################################################################################################################
#Etapa final:
# Carregar o arquivo CSV para um DataFrame
df = pd.read_csv("/content/base edt1.csv")

# Excluir a coluna 'track_genre'
if 'track_genre' in df.columns:
    df.drop(columns=['track_genre'], inplace=True)

# Converter as colunas (exceto 'explicit', 'mode' e 'time_signature') para float
for column in df.columns:
    if column not in ['explicit', 'mode', 'time_signature']:
        df[column] = pd.to_numeric(df[column], errors='coerce')

# Avaliar o algoritmo
n_folds = 5
learn_rate = 0.3
n_epochs = 50
n_codebooks = 15
scores = evaluate_algorithm(df.values.tolist(), learning_vector_quantization, n_folds, n_codebooks, learn_rate, n_epochs)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
########################################################################################################################################################################################################
#Resultado:
Foram utilizadas três distâncias para o modelo LVQ (Euclidiana, Manhattan e Chebyshev). E a melhor configuração para LVQ foi obtida ao usar a distância de Manhattan, que alcançou uma acurácia média de 89,36%.
########################################################################################################################################################################################################
