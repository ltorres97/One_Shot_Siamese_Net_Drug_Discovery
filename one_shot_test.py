# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 18:22:44 2019

@author: luist
"""
from __future__ import print_function
from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling2D
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy
import numpy.random as rng
import numpy as np
import time
import os
from Siamese_Loader import Load_Siamese
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import csv
import heapq
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit import Chem
from molvs import validate_smiles
from tensorflow import keras




def smiles_encoder(smiles):
    d = {'I': 1, '7': 2, '.': 3, 'l': 4, '8': 5, '(': 6, '[': 7, '2': 8, 'C': 9, 'O': 10, '+': 11, 'F': 12, '9': 13, 'S': 14, ')': 15, 'M': 16, '4': 17, '-': 18, 'N': 19, '1': 20, '3': 21, ']': 22, 'B': 23, 'r': 24, '#': 25, 'P': 26, '=': 27, 'H': 28, 'a': 29, '5': 30, '6': 31, 'g': 32,'@': 33,'o':34,'\\':35, '/':36, 'e':37,'A':38,'Z':39,'K':40, '%':41,'0':42,'i':43,'T':44,'c':45,'s':46,'G':47,'d':48,'n':49,'u':50,'V':51,'R':52,'b':53,'L':54}
    #,'@': 33,'o':34,'\\':35, '/':36, 'e':37,'A':38,'Z':39,'K':40, '%':41,'0':42,'i':43,'T':44,'c':45,'s':46,'G':47,'d':48
    X = np.zeros((100, 54))
#    print(smiles)
    for i, valor in enumerate(smiles):
        if(d.get(valor) == None):
            print(valor)
#        if d.get(valor) == None:
#           print(valor)
        X[i-1, int(d.get(valor))-1] = 1
    return X

def initialize_weights(shape, name=None):

    return np.random.normal(loc = 0.0, scale = 1e-2, size = shape)
#//TODO: figure out how to initialize layer biases in keras.
def initialize_bias(shape, name=None):
 
    return np.random.normal(loc = 0.5, scale = 1e-2, size = shape)

def get_siamese_model():
    
    input_shape = (100,54,1)
    # Tensores para ambos os inputs
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    
    print(left_input)
    
    # Convolutional Neural Network
    """ -> convolução com 64 filtros de 10x10;
        -> convolução com filtros ReLU (se g(x) > 0 return value else return 0, com g a função de ativação) - ReLU activation to make all negative value to zero;
        -> convolução com uma max-pooling layer (escolhe o valor mais elevado em sub-matrizes nxn bem definidas) - Pooling layer is used to reduce the spatial volume of input image after convolution;
        -> convolução com 128 filtros de 7x7 + ReLU + maxpool;
        -> convolução com 128 filtros de 2x2 + ReLU + maxpool;
        -> convolução com 256 filtros 2x2;
        -> classificação numa fully connected layer de 4096 unidades (neurónios)"""

    #build convnet to use in each siamese 'leg'
    convnet = Sequential()
    convnet.add(Conv2D(64,(10,10),activation='relu',input_shape=input_shape,
                       kernel_initializer=initialize_weights,kernel_regularizer=l2(2e-4)))
    convnet.add(MaxPooling2D())
    convnet.add(Conv2D(128,(7,7),activation='relu',
                       kernel_regularizer=l2(2e-4),kernel_initializer=initialize_weights,bias_initializer=initialize_bias))
    convnet.add(MaxPooling2D())
    convnet.add(Conv2D(128,(2,2),activation='relu',kernel_initializer=initialize_weights,kernel_regularizer=l2(2e-4),bias_initializer=initialize_bias))
    convnet.add(MaxPooling2D())
    convnet.add(Conv2D(256,(2,2),activation='relu',kernel_initializer=initialize_weights,kernel_regularizer=l2(2e-4),bias_initializer=initialize_bias))
    convnet.add(Flatten())
    convnet.add(Dense(4096,activation="sigmoid",kernel_regularizer=l2(1e-3),kernel_initializer=initialize_weights,bias_initializer=initialize_bias))
    #encode each of the two inputs into a vector with the convnet
    encoded_l = convnet(left_input)
    encoded_r = convnet(right_input)
    
    # Layer que permite obter a diferença absoluta entre os dois encodings - concatenados
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])
    
    # Dense Layer associada a uma função sigmoide que gera um similarity score
    # gera um output entre 0 e 1 que corresponde a uma probabilidade de semelhança
    prediction = Dense(1,activation='sigmoid',bias_initializer=initialize_bias)(L1_distance)
    
    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)
    #optimizer = SGD(0.0004,momentum=0.6,nesterov=True,decay=0.0003)
       
#    model_json = model.to_json()
#    with open("model.json", "w") as json_file:
#        json_file.write(model_json)
    # return the model
    return siamese_net


if __name__=='__main__':
    
    csv_file ='structure links.csv'
    txt_file ='sl.txt'
    with open(txt_file, "w") as my_output_file:
        with open(csv_file, "r") as my_input_file:
            [ my_output_file.write("$$$".join(row)+'\n') for row in csv.reader(my_input_file)]
        my_output_file.close()
    
    with open('sl.txt', 'r') as myfile:
      data = myfile.read()
    
    data = data.split("\n")
    
    D=[]
    drug_names = []
    drug_smiles = []
    drug_ids = []
    
    c = 0
    for i in data:
        d = i.split("$$$")
        D.append(d)
        if len(d)>5 & c>0:
            drug_names.append(d[1])
            drug_ids.append(d[0])
            drug_smiles.append(d[6])
        c+=1  
    
    #for drug in data:
    #print(drug_names[0:3])
    #print(drug_smiles[0:3])
    #print(drug_ids[0:3])
    
    drug_mol=[]
    #aplicando validate_smiles(drug_smiles[1406]) verifica-se que a molecula e invalida
    #remover todas as invalidas
    for i in drug_smiles:
        if validate_smiles(i) != []:
            drug_smiles.pop(drug_smiles.index(i))
            
    
    drug_smiles = list(filter(lambda a: len(a) <= 100 and len(a)>0, drug_smiles))    
#    print(validate_smiles(drug_smiles[1407]))
#    print(len(drug_smiles))
    
    l=[]
    for j in drug_smiles:
        drug_mol.append(Chem.MolFromSmiles(j))
        l.append(len(j))
       
#    print(l)
    
    fps = [FingerprintMols.FingerprintMol(x) for x in drug_mol]
    
#    prob = DataStructs.FingerprintSimilarity(fps[0],fps[1])
    n=2
    
    # experimentar diferentes métricas de semelhança 
    # Tanimoto, Dice, Cosine, Sokal, Russel, Kulczynski, McConnaughey, and Tversky.
    
    Prob = []
    Data_Final=[]
    Data_Final1=[]
    data_prov=[]
    for i in fps:
        probs = [DataStructs.FingerprintSimilarity(i,j) for j in fps]
        indexes = heapq.nlargest(n, range(len(probs)), probs.__getitem__)
#        fps = [i for j, i in enumerate(fps) if j not in indexes]
        Data1=[drug_smiles[k] for k in indexes]
        Data= [smiles_encoder(drug_smiles[k]) for k in indexes]
#        print(Data1)
        Prob.append([probs[k] for k in indexes])
        count = 0
        for i in Data1:
            count+=1
            if(i not in data_prov and count == 1):
                data_prov.append(i)
                Data_Final.append(Data)
                Data_Final1.append(Data1)
#        for i in Data1:
#            count+=1
#            if(i not in data_prov and count == 1):
#                data_prov.append(i)
#                Data_Final.append(Data)
#                Data_Final1.append(Data1)
#                print(i)
            
    data_treino, data_teste = train_test_split([i for i in Data_Final], test_size=0.25, random_state=1)
    
    data_treino = np.asarray(data_treino)
    data_teste = np.asarray(data_teste)
    
    print(data_treino.shape)
    print(data_teste.shape)
   
    siamese_net = get_siamese_model()
    print(siamese_net.summary())
    # Hyper parameters
    evaluate_every = 10 # intervalo para avaliar as one-shot tasks
    loss_every = 20
    batch_size = 10
    n_iter = 100
    N_way = 3 # how many classes for testing one-shot 
    n_val = 25 # how many one-shot tasks to validate on
    best = -1
#    siamese_net.save_weights("model.h5")
    model_path = './weights/'
    
    loader = Load_Siamese(data_treino,data_teste)
    print("Starting training process!")
    print("-------------------------------------")
    t_start = time.time()
    opt = Adam(lr = 0.00006)
    siamese_net.compile(loss="binary_crossentropy",optimizer=opt)
    
    for i in range(1, n_iter+1):
        (inputs,targets) = loader.batch_function(batch_size)
        loss = siamese_net.train_on_batch(inputs, targets)
        if i % evaluate_every == 0:
            print("\n ------------- \n")
            print("Time for {0} iterations: {1} mins".format(i, (time.time()-t_start)/60.0))
            print("Train Loss: {0}".format(loss)) 
            val_acc = loader.oneshot_test(siamese_net, N_way, n_val, verbose=True)
            #siamese_net.save_weights(os.path.join(model_path, 'weights.{}.h5'.format(i)))
            if val_acc >= best:
                print("Current best: {0}, previous best: {1}".format(val_acc, best))
                best = val_acc
                siamese_net.save_weights(os.path.join(model_path, 'weights.{}.h5'.format(i)))
        if i % loss_every == 0:
            print("iteration {}, training loss: {:.2f},".format(i,loss))
        
    print("The final best accuracy value: {0}".format(best))    
    
    siamese_net.load_weights(os.path.join(model_path, "weights.100.h5"))
    
    #ways = np.arange(1,15,2)
    ways = 3
    resume =  False
    trials = 25
    
    val_accs, train_accs,nn_accs = [], [], []
    #for N in ways:    
    val_accs.append(loader.oneshot_test(siamese_net, 3, trials, "val", verbose=True))
    train_accs.append(loader.oneshot_test(siamese_net, 3, trials, "train", verbose=True))
    #nn_acc = loader.test_nn_accuracy(N, trials,loader)
    #nn_accs.append(nn_acc)
    #print ("NN Accuracy = ", nn_acc)
    print("---------------------------------------------------------------------------------------------------------------")
        
    