# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 12:31:56 2020

@author: luist
"""
import matplotlib.pyplot as plt

#nota importante: https://machinelearningmastery.com/make-predictions-scikit-learn/
# guia para probability predictions 

x = [2, 3, 4, 5, 7, 10]#Number of possible one-shot tasks
y1 = [95, 90, 86, 78, 70, 60]#NN_val
y2 = [94, 92, 84, 86, 72, 60]#NN_train
y3 = [69.5, 55, 48.5, 42.5, 35.5, 30]#Knn_val
y4 = [61, 43, 34, 31, 22, 18.5]#Random_val
y5 = [56, 42, 24, 30, 16, 12]#SVM - classificador adaptado para devolver probabilidades
y6 = [51, 40, 23, 23, 16, 13]#Logistic Regression - classificador probabilistico
y7 = [71, 58, 60, 44, 34, 20]#Random Forest
y8 = [72, 48, 44, 43, 35, 28]#Stochastic Gradient Descendent(SGD-log) - classificador probabilistico usando a função log
y9 = [51, 32, 29, 20, 14, 13]#Naive-bayes Bernoulli - binary/boolean inputs
y10= [76, 60, 36, 34, 22, 13]#Multi-Layer Percepton NN
y11= [57, 29, 20, 29, 16, 11]#Multinomial NB
y12= [47, 35, 23, 23, 15, 6]#Complement NB - applied to text-class tasks
y13= [50, 24, 30, 18, 16, 7]#ExtraTreesCalssifier - As in random forests, a random subset of candidate features is used,
# but instead of looking for the most discriminative thresholds, thresholds are drawn at random for each 
#candidate feature and the best of these randomly-generated thresholds is picked as the splitting rule.


#y5 = [64.5, 50.8, 50, 48.6, 44.8, 44.6] #Logistic Regression
#y6 = [73.6, 64.8, 47.6, 42.8, 25.8, 21] #Random Forest
#y5 = [71, 69, 62, 55, 29, 7]#Naive-Bayes
#y6 = [53, 32.6, 27, 26.4, 16.2, 12.8] #Logistic Regression
#y7=[43.5, 24, 24, 28, 18, 9] #Random Forest
#y8 = [53, 36, 32, 13, 13, 2] #Decision Tree
#29

## Tendo em conta a probabilidade de o par pertencer à mesma classe - pred == 1
#y5 = [51, 40, 23, 23, 16, 13] #Logistic Regression
#y6 = [71, 58, 60, 44, 34, 38]#Random Forest
#y7 = [81,]#Naive-Bayes
#y8 = [56, 42, 24, 30, 16, 12]#SVM


plt.plot(x,y1, 'r-',label='NNs Val')
plt.plot(x,y2, 'b-', label = 'NNs Train')
plt.plot(x,y3, 'k-', label = 'KNN Val')
plt.plot(x,y4, 'g-', label = 'Random Val')
plt.plot(x, y5, 'y-', label = 'SVM')
plt.plot(x, y6, 'm-', label  = 'Logistic Regression')
plt.plot(x, y7, 'c-', label  = 'Random Forest')
plt.plot(x,y8, 'tab:orange', label = 'SGD-log')
plt.plot(x,y9, 'tab:purple', label = 'Bernoulli NB')
plt.plot(x,y10,'tab:brown',label = 'Multi-Layer Percepton NN')
plt.plot(x,y11,'tab:gray',label = 'Multinomial NB')
plt.plot(x,y12,'tab:pink',label = 'Complement NB')
plt.plot(x,y13,'tab:olive',label = 'ExtraTreesClassifier')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.xlabel('N')
plt.ylabel('Accuracy in %')
plt.suptitle('N-way One-Shot Learning Accuracy vs Other Models Accuracy')
plt.show()