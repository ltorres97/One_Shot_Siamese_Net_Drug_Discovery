# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 18:21:42 2019

@author: luist
"""

import numpy.random as rng
import numpy as np
import keras
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn import svm
from keras.layers import Input, Conv1D, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling1D, Dropout, MaxPooling2D
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.metrics import accuracy_score
#from sklearn.metrics import confusion_matrix

class Load_Siamese:
    
    def __init__(self,X_treino,Xval):
        self.Xval = Xval
        self.X_treino = X_treino
        self.n_classes,self.n_exemplos,self.w,self.h = X_treino.shape
        self.n_val,self.n_ex_val,_,_ = Xval.shape

    def batch_function(self,n,s='train'):
        
        if s == 'train':
            X = self.X_treino
        else:
            X = self.Xval
                   
        n_classes, n_exemplos, w, h = X.shape
    
        """Cria um batch de n pares, metade da mesma classe e a outra metade de diferentes classes para o treino da rede"""
        categorias = rng.choice(n_classes,size=(n,),replace=False)
        pares=[np.zeros((n, w, h,1)) for i in range(2)]
        targets=np.zeros((n,))
        targets[n//2:] = 1
        for i in range(n):
            categoria = categorias[i]
            index_1 = rng.randint(0,n_exemplos)
            pares[0][i,:,:,:] = X[categoria,index_1].reshape(w,h,1)
            index_2 = rng.randint(0,n_exemplos)
            #pick images of same class for 1st half, different for 2nd
            
            categoria_2 = categoria if i >= n//2 else (categoria + rng.randint(1,n_classes)) % n_classes
            pares[1][i,:,:,:] = X[categoria_2,index_2].reshape(w,h,1)
        
        return pares, targets
    
    def one_shot_task(self,N,s='val'):
        
        if s == 'train':
            X = self.X_treino
        else:
           X = self.Xval
           
        n_classes, n_exemplos, w, h = X.shape   
        """Cria pares de uma imagem teste e de um outro conjunto para testar a rede - N-way one-shot learning"""
        categorias = rng.choice(n_classes,size=(N,),replace=False)
        indexes = rng.randint(0,n_exemplos,size=(N,))
        true_categoria = categorias[0]
        exemplo_1, exemplo_2 = rng.choice(n_exemplos,replace=False,size=(2,))
        imagem_teste = np.asarray([X[true_categoria,exemplo_1,:,:]]*N).reshape(N,w,h,1)
        support_set = X[categorias,indexes,:,:]
        support_set[0,:,:] = X[true_categoria,exemplo_2]
        support_set = support_set.reshape(N,w,h,1)
        pares = [imagem_teste,support_set]
        targets = np.zeros((N,))
        targets[0] = 1
        
       # print("Pares:",pares)
        return pares, targets
    
    def oneshot_test(self,model,N,k,s = "val"):
        
        """Avalia a accuracy média da rede na determinação da classe das imagens ao longo de um numero k de tasks"""
        n_corretos = 0
    #        if verbose:
    #            print("Evaluating model on {} unique {} way one-shot learning tasks ...".format(k,N))
        for i in range(k):
            inputs, targets = self.one_shot_task(N,s)
            probs = model.predict(inputs)
            if np.argmax(probs) == 0:
                n_corretos+=1
        percent_correct = (n_corretos / k)
    #        if verbose:
    #            print("Got an average of {}% {} way one-shot learning accuracy".format(percent_correct,N))
        return percent_correct
        
        
    def testing(self, N, trials, tam, s= 'val'):
        
        pairs_train, targets_train = self.batch_function(tam)
        
        lista1 = pairs_train[0]
        lista2 = pairs_train[1]

#        print(lista1[0])
        
        pairs_train2 = []
        
        for i in range(len(lista1)):
            seq3=[]
            seq = lista1[i].flatten()
            seq2 = lista2[i].flatten()
#            print(seq.shape)
#            print(seq2.shape)
            for j in seq:
                seq3.append(j)
            for k in seq2:
                seq3.append(k)
            
            pairs_train2.append(np.asarray(seq3))
        
        n_corretos = 0
        pairs2train=np.asarray(pairs_train2).reshape(tam,54*100*2,1)
        targets_train = np.asarray(targets_train).reshape(tam,1)
        print(pairs2train.shape)
        print(targets_train.shape)
        n_timesteps, n_features, n_outputs = pairs2train.shape[1], pairs2train.shape[2], targets_train.shape[1]
        
        cnn_net = self.cnn_model(pairs2train, targets_train,n_timesteps, n_features, n_outputs)
#        model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
        
#        cnn_net= model.fit(pairs2train, targets_train, batch_size=50,epochs=20,verbose=1)
        
        lista_acc=[]
        for n in range(2,N+1):
            for t in range(trials):
            
                print(t)
                pairs_val2=[]
                pairs_val,targets_val = self.one_shot_task(n,s)
                lista11= pairs_val[0]
                lista22 = pairs_val[1]
                
                for i2 in range(len(lista11)):
                
                    seq3=[]
                    seq = lista11[i2].flatten()
                    seq2 = lista22[i2].flatten()
        #            print(seq.shape)
        #            print(seq2.shape)
                    for j2 in seq:
                        seq3.append(j2)
                    for k2 in seq2:
                        seq3.append(k2)
                        
                    pairs_val2.append(np.asarray(seq3))
                                
                pairs2val=np.asarray(pairs_val2).reshape(n,54*100*2,1)
    #            print(pairs2train.shape)
    #            print(pairs2val.shape)
    #            print(targets_train.shape)
    #            print(targets_train)
                
    #            kernel = 1.0 * RBF(1.0)
    #            reg= SGDClassifier(loss="log", penalty="l2", max_iter=5)
    #            reg = GaussianProcessClassifier(kernel=kernel,random_state=0)
    #            reg = LinearRegression().fit(pairs2train, targets_train)
    #            reg = GaussianNB()
    #            reg = ComplementNB()
    #            reg = MultinomialNB()
    #            reg= ExtraTreesClassifier(n_estimators=100, random_state=0)
    #            reg = BernoulliNB()
    #            reg = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10,5, 2), random_state=1)
    #            reg = LogisticRegression().fit(pairs2train,targets_train)
    #            reg = RandomForestClassifier(max_depth=2, random_state=0)
    #            reg = tree.DecisionTreeClassifier()
    #            reg = svm.SVC(probability = True)
    #                
    #            reg.fit(pairs2train, targets_train)
                targets_val = np.asarray(targets_val).reshape(n,1)
    #            reg = cnn_net.evaluate(pairs2val,targets_val,verbose=0)
    #            print('Test loss:',reg[0])
    #            print('Test accuracy:', reg[1])
                print(pairs2val.shape)
                print(targets_val.shape)
                
    #            val1 = reg.predict(pairs2val)
                val2 = cnn_net.predict(pairs2val)
                
    #                val = reg3.predict(pairs2val)'
        #            val2 = reg2.predict(pairs2val)
                
                print("Target:",targets_val)
    #            print("Previsão:",val1)
                print("Previsão Probabilidade:",val2)
    #            print("Val[1:]", val[1:])
                pred=[]
                for i in val2:
                    pred.append(i[0])
                
                if np.argmax(pred) == 0:
                    n_corretos +=1
    #            verifica = np.asarray(val[1:])
    #            
    ##            verifica = val[1:]
    #            if all(verifica == 0) and val[0] == 1:
    ##            if list(pred) == list(targets_val): #o que significa que o primeiro argumento corresponde ao máximo da lista
    #                n_corretos +=1
                        
            percent_correct = (n_corretos / trials)
            
            lista_acc.append(percent_correct)
            n_corretos= 0
        
        print(lista_acc)
        return lista_acc
      
        
    def knn_test(self, N, trials, s='val'):
                
        n_corretos = 0 
        
        for i in range(trials):
            pairs, targets = self.one_shot_task(N,s)
            correto = self.knn_prediction(pairs,targets)
            n_corretos += correto
        
        return n_corretos/ trials
    
    
    def random_test(self, N, trials, s='val'):
        
        n_corretos = 0 
        
        for i in range(trials):
            pairs, targets = self.one_shot_task(N,s)
            correto = self.random_prediction(pairs,targets)
            n_corretos += correto
        
        return n_corretos/ trials
    
        
    def knn_prediction(self,pairs,targets):
        """returns 1 if nearest neighbour gets the correct answer for a one-shot task given by (pairs, targets)"""
        L2_distances = np.zeros_like(targets)
#        print("PARES NN:", pairs[0][1].flatten())
        #print("PARES NN2:", pairs[1][1])
        for i in range(len(targets)):
            L2_distances[i] = np.sqrt(np.sum((pairs[0][i].flatten() - pairs[1][i].flatten())**2))
        if np.argmin(L2_distances) == np.argmax(targets):
            return 1
        return 0
    
    
    def random_prediction(self,pairs,targets):
        
        random_predictions = np.zeros_like(targets)
        
        for i in range(len(targets)):
            random_predictions[i] = np.random.uniform(0,1)
        if np.argmax(random_predictions) == np.argmax(targets):
            return 1
        return 0

    def cnn_model(self,trainX, trainy,n_timesteps, n_features, n_outputs):
        
        verbose, epochs, batch_size = 1, 10, 50
#        input_shape = (10000,1)
#        input_net = Input(input_shape)
        conv_model = Sequential()
        conv_model.add(Conv1D(32,10,activation='relu', input_shape=(n_timesteps,n_features)))
        conv_model.add(MaxPooling1D())
        conv_model.add(Conv1D(64,8,activation='relu', input_shape=(n_timesteps,n_features)))
        conv_model.add(MaxPooling1D())
        conv_model.add(Conv1D(128,4,activation='relu'))
        conv_model.add(MaxPooling1D())
        conv_model.add(Conv1D(256,2,activation='relu'))
        conv_model.add(MaxPooling1D())
        conv_model.add(Dropout(0.5))
        conv_model.add(MaxPooling1D())
        conv_model.add(Flatten())
        conv_model.add(Dense(100,activation="relu"))
        conv_model.add(Dense(n_outputs, activation='sigmoid'))
        
        conv_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        conv_model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
        
        return conv_model
    
    # n=2 - 0.78
    # n=3 - 0.42
    # n=5 - 0.26
    # n=7 -     
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
#   
#
#    def knn(self,k,N,s='val'):
#        
#        knn = KNeighborsClassifier()
#        pairs,targets = self.batch_function(n)
#        
#        knn.fit(pairs,targets)
#    
#        n_right = 0
#        
#        for i in range(k):
#            pairs_test,targets_test = self.one_shot_task(N,s)
#            pred = knn.predict(pairs_test)
#            if np.argmin(pred) == np.argmax(targets):
#                n_right += correct
#        
#        return 100.0 * n_right / k
#    
        
        
#    def nearest_neighbour_correct2(self,pairs,targets,n,k):
#        
#       knn = KNeighborsClassifier(n_neighbors=3)
#       pairs, targets = self.batch_function(n)
#       knn.fit(pairs, targets)
#       
#       n_right = 0
#        
#       for i in range(k):
#           pairs, targets = self.one_shot_task(N,s)
#           pred = knn.predict(pairs)
#           correct = self.nearest_neighbour_correct2(pairs,targets)
#           n_right += correct
#        
#        return 100.0 * n_right / k
    
       
#    def test_nn_accuracy(self,N_ways,n_trials,loader):
#        """Returns accuracy of one shot """
#        print("Evaluating nearest neighbour on {} unique {} way one-shot learning tasks ...".format(n_trials,N_ways))
#    
#        n_right = 0
#        for i in range(n_trials):
#            pairs,targets = self.one_shot_task(N_ways,"val")
#            correct = self.nearest_neighbour_correct(pairs,targets)
#            n_right += correct
#            
#        return (100.000 * n_right / n_trials)