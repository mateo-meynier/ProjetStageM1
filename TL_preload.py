#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import sklearn as skl
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import re
import sys
from math import floor
import matplotlib.pyplot as plt
from Bio import SeqIO


# In[2]:


def convert_seq_to_array(seq) :
    seq = seq.lower()
    seq = re.sub('[^acgt]','z',seq)
    seq_array = np.array(list(seq))
    return seq_array

def one_hot_encoder(seq_array) :
    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse = False, dtype=int,categories=[range(5)])
    seq_convert_int = label_encoder.fit_transform(seq_array)
    seq_convert_int = seq_convert_int.reshape(len(seq_convert_int),1)
    seq_onehot = onehot_encoder.fit_transform(seq_convert_int)
    seq_onehot = np.delete(seq_onehot, -1, 1)
    return seq_onehot


# In[3]:


def saveDataSet(path,train_genenames, train_expressions, train_sequences, test_genenames, test_expressions, test_sequences) :
    
    train_genenames = np.array(train_genenames)
    train_sequences = np.array(train_sequences)
    train_expressions = np.array(train_expressions)
    
    #variance_y = np.var(train_expressions)
    #moyenne_y = np.mean(train_expressions)
    #train_expressionsnorm = (train_expressions-moyenne_y)/variance_y
    
    print('before',train_expressions)
    #train_expressions = train_expressions.reshape(-1,1)
    #dataset_scaler = StandardScaler()
    #train_expressions = dataset_scaler.fit_transform(train_expressions)
    #train_expressions = train_expressions.reshape(-1)
    #print('after',train_expressionsnorm)

    np.save(path + "genenames_trainset.npy", train_genenames)
    np.save(path + "seqs_trainset.npy", train_sequences)
    np.save(path + "exps_trainset.npy", train_expressions)
    print("Enregistrement du training set fini...\n")
    
    test_genenames = np.array(test_genenames)
    test_sequences = np.array(test_sequences)
    test_expressions = np.array(test_expressions)
    
    #variance_y = np.var(test_expressions)
    #moyenne_y = np.mean(test_expressions)
    #test_expressionsnorm = (test_expressions-moyenne_y)/variance_y
    
    print("before",test_expressions)
    #test_expressions = test_expressions.reshape(-1,1)
    #test_expressions = dataset_scaler.transform(test_expressions)
    #test_expressions = test_expressions.reshape(-1)
    #print("after",test_expressionsnorm)

    np.save(path + "genenames_testset.npy", test_genenames)
    np.save(path + "seqs_testset.npy", test_sequences)
    np.save(path + "exps_testset.npy", test_expressions)
    print("Enregistrement du test set fini...\n")


# In[4]:


def saveClusterDataSet(path,ctrain_genenames, ctrain_sequences, ctrain_expressions ,ctest_genenames,ctest_sequences,ctest_expressions,cluster):
    
    ctrain_genenames = np.array(ctrain_genenames)
    ctrain_sequences = np.array(ctrain_sequences)
    ctrain_expressions = np.array(ctrain_expressions)
    
    #variance_y = np.var(ctrain_expressions)
    #moyenne_y = np.mean(ctrain_expressions)
    #ctrain_expressionsnorm = (ctrain_expressions-moyenne_y)/variance_y
    
    print("before",ctrain_expressions)
    #ctrain_expressions = ctrain_expressions.reshape(-1,1)
    #cluster_scaler = StandardScaler()
    #ctrain_expressions = cluster_scaler.fit_transform(ctrain_expressions)
    #ctrain_expressions = ctrain_expressions.reshape(-1)
    #print("after",ctrain_expressionsnorm)

    np.save(path + "cluster"+ str(cluster)+"_genenames_trainset.npy", ctrain_genenames)
    np.save(path + "cluster"+ str(cluster)+"_seqs_trainset.npy", ctrain_sequences)
    np.save(path + "cluster"+ str(cluster)+"_exps_trainset.npy", ctrain_expressions)
    print("Enregistrement du training set", cluster," fini...\n")
    
    ctest_genenames = np.array(ctest_genenames)
    ctest_sequences = np.array(ctest_sequences)
    ctest_expressions = np.array(ctest_expressions)
    
    #variance_y = np.var(ctest_expressions)
    #moyenne_y = np.mean(ctest_expressions)
    #ctest_expressionsnorm = (ctest_expressions-moyenne_y)/variance_y  
    
    print("before",ctest_expressions)
    
    #ctest_expressions = ctest_expressions.reshape(-1,1)
    #ctest_expressions = cluster_scaler.transform(ctest_expressions)
    #ctest_expressions= ctest_expressions.reshape(-1)
    #print("after",ctest_expressionsnorm)

    np.save(path + "cluster"+ str(cluster)+"_genenames_testset.npy", ctest_genenames)
    np.save(path + "cluster"+ str(cluster)+"_seqs_testset.npy", ctest_sequences)
    np.save(path + "cluster"+ str(cluster)+"_exps_testset.npy", ctest_expressions)
    print("Enregistrement du test set", cluster," fini...\n")


# In[5]:


def saveDataSetlog(path,train_genenames, train_expressions, train_sequences, test_genenames, test_expressions, test_sequences) :
    
    train_genenames = np.array(train_genenames)
    train_sequences = np.array(train_sequences)
    train_expressions = np.array(train_expressions)

    
    print('before',train_expressions)
    train_expressionslog = np.log(train_expressions)
    print('after',train_expressionslog)


    np.save(path + "genenames_trainset.npy", train_genenames)
    np.save(path + "seqs_trainset.npy", train_sequences)
    np.save(path + "exps_trainset.npy", train_expressionslog)
    print("Enregistrement du training set fini...\n")
    
    test_genenames = np.array(test_genenames)
    test_sequences = np.array(test_sequences)
    test_expressions = np.array(test_expressions)

    
    print("before",test_expressions)
    test_expressionslog = np.log(test_expressions)

    print("after",test_expressionslog)

    np.save(path + "genenames_testset.npy", test_genenames)
    np.save(path + "seqs_testset.npy", test_sequences)
    np.save(path + "exps_testset.npy", test_expressionslog)
    print("Enregistrement du test set fini...\n")


# In[6]:


def saveClusterDataSetlog(path,ctrain_genenames, ctrain_sequences, ctrain_expressions ,ctest_genenames,ctest_sequences,ctest_expressions,cluster):
    
    ctrain_genenames = np.array(ctrain_genenames)
    ctrain_sequences = np.array(ctrain_sequences)
    ctrain_expressions = np.array(ctrain_expressions)

    
    ctrain_expressionslog = np.log(ctrain_expressions)
    print(ctrain_expressions)
    print(ctrain_expressionslog)

    np.save(path + "cluster"+ str(cluster)+"_genenames_trainset.npy", ctrain_genenames)
    np.save(path + "cluster"+ str(cluster)+"_seqs_trainset.npy", ctrain_sequences)
    np.save(path + "cluster"+ str(cluster)+"_exps_trainset.npy", ctrain_expressionslog)
    print("Enregistrement du training set", cluster," fini...\n")
    
    ctest_genenames = np.array(ctest_genenames)
    ctest_sequences = np.array(ctest_sequences)
    ctest_expressions = np.array(ctest_expressions)
 
    
    print("before",ctest_expressions)
    
    ctest_expressionslog = np.log(ctest_expressions)
    
    print("after",ctest_expressionslog)

    np.save(path + "cluster"+ str(cluster)+"_genenames_testset.npy", ctest_genenames)
    np.save(path + "cluster"+ str(cluster)+"_seqs_testset.npy", ctest_sequences)
    np.save(path + "cluster"+ str(cluster)+"_exps_testset.npy", ctest_expressionslog)
    print("Enregistrement du test set", cluster," fini...\n")


# In[7]:


#%%time
clusters = []
with open('cluster_101bp_sumQ20.fa') as fasta_file:
    for seq_record in SeqIO.parse(fasta_file, 'fasta'):
        split_id = seq_record.id.split("|")
        clusters.append(split_id[1])
print("Nbr exemples :",len(clusters))


# In[8]:


clusters = list(set(clusters)) #rendre la liste unique
print("Clusters :",clusters)
print("nombre Clusters :",len(clusters))


# In[9]:


#%%time
cluster_list = []
for i in range(len(clusters)+1) :
    cluster_list.append([])
    
with open('cluster_101bp_sumQ20.fa') as fasta_file:
    for seq_record in SeqIO.parse(fasta_file, 'fasta'):
        split_id = seq_record.id.split("|")
        cluster = int(split_id[1])
        cluster_list[cluster].append(seq_record)


# In[10]:


#%%time
path = "./cluster_dataset/"
i=0
ratio = 0.20
data_train = []
data_test = []
for i in range(1,len(cluster_list)) :
    cluster=[]
    for seq_record in cluster_list[i] :
        cluster.append(seq_record)
        
    np.random.shuffle(cluster)
    split = int(np.floor(ratio * len(cluster)))
    print("Split indice c",i,":",split)
    cluster_train, cluster_test = cluster[split:], cluster[:split]
    print("cluster_train :",len(cluster_train), "/","cluster_test :", len(cluster_test))
     
    data_train.extend(cluster_train)
    data_test.extend(cluster_test)


    ctrain_genenames = []
    ctrain_sequences = []
    ctrain_expressions = []
    
    for seq_record in cluster_train :
        #data_train.append(seq_record)
        split_id = seq_record.id.split("|")
        
        ctrain_genenames.append(split_id[0])
        ctrain_expressions.append(float(split_id[2]))
        sequence = str(seq_record.seq)
        seqconvert = one_hot_encoder(convert_seq_to_array(sequence))
        ctrain_sequences.append(seqconvert)
    print("Len cluster trainset :",len(ctrain_sequences))
    
    ctest_genenames = []
    ctest_sequences = []
    ctest_expressions = []
    for seq_record in cluster_test :
        #data_test.append(seq_record)
        split_id = seq_record.id.split("|")
        
        ctest_genenames.append(split_id[0])
        ctest_expressions.append(float(split_id[2]))
        sequence = str(seq_record.seq)
        seqconvert = one_hot_encoder(convert_seq_to_array(sequence))
        ctest_sequences.append(seqconvert)
    print("Len cluster testset :",len(ctest_sequences))

    
    saveClusterDataSetlog(path,ctrain_genenames, ctrain_sequences, ctrain_expressions ,ctest_genenames,ctest_sequences,ctest_expressions,i)
    i+=1


# In[11]:


#%%time
path = "./dataset/"
print("data_train :",len(data_train), "/","data_test :", len(data_test))
train_genenames = []
train_expressions = []
train_sequences = []

test_genenames = []
test_expressions = []
test_sequences = []

for seq_record in data_train :
        split_id = seq_record.id.split("|")
        train_genenames.append(split_id[0])
        train_expressions.append(float(split_id[2]))
        seq = str(seq_record.seq)
        seqconvert = one_hot_encoder(convert_seq_to_array(seq))
        train_sequences.append(seqconvert)

for seq_record in data_test :
        split_id = seq_record.id.split("|")
        test_genenames.append(split_id[0])
        test_expressions.append(float(split_id[2]))
        seq = str(seq_record.seq)
        seqconvert = one_hot_encoder(convert_seq_to_array(seq))
        test_sequences.append(seqconvert)
print("train_sequences :",len(train_sequences), "/","test_sequences :", len(test_sequences))

saveDataSetlog(path,train_genenames, train_expressions, train_sequences, test_genenames, test_expressions, test_sequences)

