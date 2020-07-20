#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sklearn as skl
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import re
import sys
from math import floor
import matplotlib.pyplot as plt
from Bio import SeqIO


# parserSequencesList : fct séparant les élements de la liste dans 3 autres listes et convertit les séquences en encodage one hot

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

def parserSequencesList(sequences_list) :
    genenames = []
    seqs =[]
    signals = []
    lenseqs = []
    for element in sequences_list :
        genenames.append(element[0])
        signals.append(element[1])
        seqconvert = one_hot_encoder(convert_seq_to_array(element[2]))
        seqs.append(seqconvert)
        
    return genenames, seqs, signals


# saveDataSet : fonction sauvegardant les deux sets de données : le set d'entrainement et le set de training

# In[3]:


def saveDataSet(path_to_save,genenames_train, seqs_train, signals_train, genenames_test, seqs_test, signals_test) :
    
    print("Debut de l'enregistrement des deux datasets...")

    genenames_train = np.array(genenames_train)
    seqs_train = np.array(seqs_train)
    signals_train = np.array(signals_train)
    
    genenames_test = np.array(genenames_test)
    seqs_test = np.array(seqs_test)
    signals_test = np.array(signals_test)
    
    print("Sauvegarde des deux sets ....")

    np.save(path + "genenames_trainset.npy", genenames_train)
    np.save(path + "seqs_trainset.npy", seqs_train)
    np.save(path + "signals_trainset.npy", signals_train)
    print("Enregistrement du training set fini...")

    np.save(path + "genenames_testset.npy", genenames_test)
    np.save(path + "seqs_testset.npy", seqs_test)
    np.save(path + "signals_testset.npy", signals_test)

    print("Enregistrement du test set fini...")


# In[4]:


with open('geneName_seq_all.fa') as fasta_file:
    genenames = []
    sequences = []
    for seq_record in SeqIO.parse(fasta_file, 'fasta'):
        genenames.append(seq_record.id)
        sequences.append(str(seq_record.seq))


# In[5]:


df1 = pd.DataFrame(dict(gene=genenames, Sequence=sequences))
df1


# In[6]:


df2 = pd.read_csv('GeneName_sig_all.csv', low_memory = False, sep = ',', header = 0 )
df2.rename(columns={'gene': 'gene', ' sig ': 'sig'}, inplace=True)
df2


# In[7]:


df= pd.merge(df2, df1, on='gene', how='inner')
df = df.drop(df[df.sig == 2].index)
#suppression des lignes avec signal = 2
sig1 = len(df[df.sig == 1])
sig0 = len(df[df.sig == 0])

percsig1 = (sig1 / (sig1+sig0)) *100
percsig0 = (sig0 / (sig1+sig0)) *100
print(" --- SIG1 : ",percsig1,"% ---","(",sig1,")")
print(" --- SIG0 : ",percsig0,"% ---","(",sig0,")")

plt.bar(["0","1"], [sig0,sig1], width = 0.5, color = 'blue')
df


# In[8]:


# bloc à enlever pour utiliser toutes les données
dfsig1 = df[df.sig == 1].index
lensig2 = len(dfsig1)*0.8
df = df.drop(dfsig1[0:int(lensig2)])


# In[9]:


sig1 = len(df[df.sig == 1])
sig0 = len(df[df.sig == 0])
percsig1 = (sig1 / (sig1+sig0)) *100
percsig0 = (sig0 / (sig1+sig0)) *100
print(" --- SIG1 : ",percsig1, "% ---")
print(" --- SIG0 : ",percsig0, "% ---")

print(" --- SIG1 number : ",sig1,  "---")
print(" --- SIG0 number : ",sig0, " ---")
plt.bar(["0","1"], [sig0,sig1], width = 0.5, color = 'blue')
plt.show()


# In[10]:


mycolumns = ['gene']
dfgene = df[mycolumns]
dfgene


# In[11]:


sequences_list = df.values.tolist()
len(sequences_list)


# ### création du set de test en récupérant les noms de 20% des gènes

# In[12]:


#%%time
cpt = 0
allgenelist = []
genenamelist = []
for row in dfgene.itertuples():
    gene = row.gene
    allgenelist.append(gene)
    genesplit = gene.split("_")
    genenamelist.append(genesplit[0])
print(len(genenamelist))
genenamelist = list(set(genenamelist)) #rendre la liste unique
print("nbr de gènes",len(genenamelist))
test_genes = random.sample(genenamelist, int(0.20 * len(genenamelist)))

with open('testset_names.txt', 'w') as out_file:
    for gene in allgenelist:
        genesplit = gene.split("_")
        if genesplit[0] in test_genes:
            out_file.write(str(gene) + '\n')


# ### lecture du fichier contenant le nom des gènes pour le testset

# In[13]:


data = pd.read_csv('testset_names.txt', header = None)
data


# In[14]:


datalist = data.values.tolist()


# ### récupération de la ligne (gène, signal, sequenceNNN) pour chacun des gènes de datalist

# In[15]:


#%%time
testset_list = []
cpt = 0
for gene in datalist :
    ligne = df.loc[df['gene'] == gene[0]]
    
    if (ligne.empty) : 
        cpt += 1
    else :
        #df = df.drop(ligne.index)
        lignelist = ligne.values.tolist()
        lignelist = lignelist[0]
        testset_list.append(lignelist)
print(testset_list)
print("Compteur ligne vide :", cpt)


# ### ajout des lignes n'apparaissant pas dans testset_list dans la liste trainset

# In[16]:


#%%time
trainset_list = []
for i in range(len(sequences_list)):
    if sequences_list[i] not in testset_list :
        trainset_list.append(sequences_list[i])
print(trainset_list)


# In[17]:


print(len(trainset_list))
print(len(testset_list))
print(len(sequences_list))


# ### division des deux listes et appel des fonctions de sauvegarde

# In[19]:


random.shuffle(trainset_list)
random.shuffle(testset_list)
genenames_train, seqs_train, signals_train =  parserSequencesList(trainset_list)
genenames_test, seqs_test, signals_test = parserSequencesList(testset_list)

print(type(signals_train[0]))
print(type(seqs_train[0]))
#path = "./dataset/"
# pour tester sur des un set de données avec nbr classes signaux équivalentes
path = "./datasetsplit/"
saveDataSet(path,genenames_train, seqs_train, signals_train, genenames_test, seqs_test, signals_test)



