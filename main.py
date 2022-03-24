import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#DIMINUI ESCALA E AUMENTA ESPAÇO (ACO QUE DIMINIU A PORCENTAGEM)
#TESTE COM E SEM PCA
data = load_iris()

features =data.data
target = data.target


plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
plt.scatter(features[:,0], features[:,1], c=target,marker='o',cmap='viridis')

Classificador = MLPClassifier(hidden_layer_sizes = (10), alpha=1, max_iter=100)
#hidden_layer_sizes NEURNIO
Classificador.fit(features,target)
predicao = Classificador.predict(features)

plt.subplot(2,2,3)
plt.scatter(features[:,2], features[:,3], c=predicao,marker='d',cmap='viridis',s=150)
plt.scatter(features[:,2], features[:,3], c=target,marker='o',cmap='viridis',s=15)


pca = PCA(n_components=2, whiten=True, svd_solver='randomized')
pca = pca.fit(features)
pca_features = pca.transform(features)
print('Mantida %5.2f%% da informação do conjunto inicial de dados'%(sum(pca.explained_variance_ratio_)*100))

plt.subplot(2,2,2)
plt.scatter(pca_features[:,0], pca_features[:,1], c=target,marker='o',cmap='viridis')#ESQ
#PROBLEMA TEM 4 FEATURES, ESA SENDO PLOTADO 2 (NAO DA PR APLOTAR AS 4)
#pca  PRINC COMP ANALI



Classificador = MLPClassifier(hidden_layer_sizes = (10), alpha=1, max_iter=100)
Classificador.fit(features,target)


predicao = Classificador.predict(features)

plt.subplot(2,2,4)
plt.scatter(features[:,2], features[:,3], c=predicao,marker='d',cmap='viridis',s=150)#DIR
plt.scatter(features[:,2], features[:,3], c=target,marker='o',cmap='viridis',s=15)#DIR
#PONTO NO LOS NA MESMA COR ESTA BOM/ PONO NO LOS COR DIFERENTE CLAS TA RUIM (REFAZER)
plt.show()


plot_confusion_matrix(Classificador, features, target,include_values=True,display_labels=data.target_names)
plt.show()

