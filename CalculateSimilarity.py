import numpy as np
import pandas as pd
import gensim
import networkx as nx
import pickle as pkl
from sklearn.neighbors import KDTree
from multiprocessing import Pool, cpu_count
import time
from Bio import SeqIO
import os
from Bio import Align
def segment(seq):
    res = []
    i = 0
    while i + 3 < len(seq):
        tmp = seq[i:i + 3]
        res.append(tmp)
        i = i + 1
    return res


def read_fa(path):
    res = {}
    rescords = list(SeqIO.parse(path, format="fasta"))
    for x in rescords:
        id = str(x.id)
        seq = str(x.seq).replace("U", "T").replace("N", "")
        res[id] = seq
    return res


def save_dict(x_dict, path):
    f = open(path, "w")
    for k, v in x_dict.items():
        tmp = k + "," + ",".join([str(x) for x in v])
        f.write(tmp + "\n")
    f.close()


def load_dict(path):
    lines = open(path, "r").readlines()
    res = {}
    for line in lines:
        x_list = line.strip().split(",")
        id = str(x_list[0])
        vec = [np.float(x) for x in x_list[1:]]
        res[id] = vec
    return res



def get_LNS(feature_matrix, neighbor_num):
    feature_matrix = np.matrix(feature_matrix)
    iteration_max = 40  # same as 2018 bibm
    mu = 3  # same as 2018 bibm
    X = feature_matrix
    alpha = np.power(X, 2).sum(axis=1)
    distance_matrix = np.sqrt(alpha + alpha.T - 2 * X * X.T)
    print(distance_matrix)
    row_num = X.shape[0]
    e = np.ones((row_num, 1))
    distance_matrix = np.array(distance_matrix + np.diag(np.diag(e * e.T * np.inf)))
    sort_index = np.argsort(distance_matrix, kind='mergesort')
    nearest_neighbor_index = sort_index[:, :neighbor_num].flatten()
    nearest_neighbor_matrix = np.zeros((row_num, row_num))
    nearest_neighbor_matrix[np.arange(row_num).repeat(neighbor_num), nearest_neighbor_index] = 1
    C = nearest_neighbor_matrix
    np.random.seed(0)
    W = np.mat(np.random.rand(row_num, row_num), dtype=float)
    W = np.multiply(C, W)
    lamda = mu * e
    P = X * X.T + lamda * e.T
    for q in range(iteration_max):
        Q = W * P
        W = np.multiply(W, P) / Q
        W = np.nan_to_num(W)
    return np.array(W)



def mirna_role2vec_embedding_x(mirna_id_vec_dict):
    mirna_list = list(mirna_id_seq_dict.keys())
    mirna_lns = get_LNS(np.array(list(mirna_id_vec_dict.values())), neighbor_num=5)
    mirna_lns[mirna_lns > 0] = 1
    g = nx.Graph()
    for i in range(len(mirna_lns)):
        for j in range(len(mirna_lns)):
            if mirna_lns[i, j] == 1:
                g.add_edge(mirna_list[i], mirna_list[j])

    f = open("../Basic data/m-m.csv", "w")
    f.write("id1,id2\n")
    for (u, v) in g.edges:
        f.write(str(u) + "," + str(v) + "\n")


def lncrna_role2vec_embedding_x(lncrna_id_vec_dict):
    lncrna_list = list(lncrna_id_vec_dict.keys())
    lncrna_lns = get_LNS(np.array(list(lncrna_id_vec_dict.values())), neighbor_num=5)
    lncrna_lns[lncrna_lns > 0] = 1
    g = nx.Graph()
    for i in range(len(lncrna_lns)):
        for j in range(len(lncrna_lns)):
            if lncrna_lns[i, j] == 1:
                g.add_edge(lncrna_list[i], lncrna_list[j])

    f = open("../Basic data/lnc-lnc.csv", "w")
    f.write("id1,id2\n")
    for (u, v) in g.edges:
        f.write(str(u) + "," + str(v) + "\n")


def load_dict(path):
    res = {}
    lines = open(path).readlines()
    for line in lines:
        x_list = line.strip().split(",")
        name = x_list[0]
        vec = [np.float(x) for x in x_list[1:]]
        res[name] = vec
    return res



mirna_id_seq_dict = read_fa("../Basic data/mirna.fa")
mirna_kmer = load_dict("../Extract sequence features/mirna_kmer.dict")
mirna_ctd = load_dict("../Extract sequence features/mirna_ctd.dict")
#mirna_miskmer = load_dict('../Extract sequence features/mrna_misker.dict')
mirna_vec = {k: mirna_kmer[k] + mirna_ctd[k] for k in list(mirna_kmer.keys())}
mirna_role2vec_embedding_x(mirna_vec)

lncrna_id_seq_dict = read_fa("../Basic data/lncrna.fa")
lncrna_kmer = load_dict("../Extract sequence features/lncrna_kmer.dict")
lncrna_ctd = load_dict("../Extract sequence features/lncrna_kmer.dict")
#lncrna_miskmer = load_dict('../Extract sequence features/lncrna_misker.dict')
lncrna_vec = {k: lncrna_kmer[k] + lncrna_ctd[k] for k in list(lncrna_kmer.keys())}
lncrna_role2vec_embedding_x(lncrna_vec)





