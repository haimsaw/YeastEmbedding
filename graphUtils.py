import networkx as nx
import itertools
import collections
from scipy.optimize import curve_fit
import numpy as np
import random
import copy
import csv

def f(k, c, m):
	return m * k ** -c

def find_cliques_size_k(G, k):
    i = 0
    for clique in nx.enumerate_all_cliques(G):
        if len(clique) == k:
            i += 1
        if len(clique) > k:
            break
    return i

def find_special_motif(G,modelNumOfNeighbors, modelNeighborsLists):
    i = 0
    for clique in nx.enumerate_all_cliques(G):
        if len(clique) == 3:
            i += len([neigh for neigh in modelNeighborsLists[clique[0]] if ((neigh not in clique) and (neigh in modelNeighborsLists[clique[1]]) and (neigh not in modelNeighborsLists[clique[2]]))])
            i += len([neigh for neigh in modelNeighborsLists[clique[0]] if ((neigh not in clique) and (neigh in modelNeighborsLists[clique[2]]) and (neigh not in modelNeighborsLists[clique[1]]))])
            i += len([neigh for neigh in modelNeighborsLists[clique[1]] if ((neigh not in clique) and (neigh in modelNeighborsLists[clique[2]]) and (neigh not in modelNeighborsLists[clique[0]]))])
        if len(clique) > 3:
            break
    return round(i / 2)

def calcClusteCoeff(modelNumOfNeighbors, modelNeighborsLists):
    # Calculate clustering coefficient
    numOfNodes = len(modelNumOfNeighbors.keys())
    cc = 0
    for node in modelNumOfNeighbors.keys():
        neighbors = modelNeighborsLists[node]
        degree = modelNumOfNeighbors[node]
        if degree == 0 or degree == 1:
            continue
        numOfConnectedNeighbors = 0
        for neigbor_i in neighbors:
            for neigbor_j in neighbors:
                if neigbor_i == neigbor_j:
                    continue
                if neigbor_i in  modelNeighborsLists[neigbor_j]:
                    numOfConnectedNeighbors += 1
        numOfConnectedNeighbors = numOfConnectedNeighbors / 2
        cc += numOfConnectedNeighbors / (degree * (degree - 1) / 2)

    return cc / numOfNodes

def calcDistParams(modelNumOfNeighbors):
    numOfNodes = len(modelNumOfNeighbors.keys())
    degreeList = list(modelNumOfNeighbors.values())
    degreeDic = {}
    for degree in degreeList:
        if degreeDic.get(degree) is None:
            degreeDic[degree] = 1
        else:
            degreeDic[degree] += 1

    degreeDist = {}
    for degree in degreeDic:
            degreeDist[degree] = degreeDic[degree] / numOfNodes

    sortedDic = collections.OrderedDict(sorted(degreeDist.items()))
    x = list(sortedDic.keys())[1:]
    y = list(sortedDic.values())[1:]
    params, _ = curve_fit(f, x, y, p0 = np.asarray([1.5, 1]))
    c1, c2  = params
    x_fit = np.arange(min(x), max(x), 1)
    y_fit = f(x_fit, c1, c2)
    return x, y, x_fit, y_fit, c1, c2

def generateEdgesListFromFile(fileName):
    tsv_file = open(fileName)
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    listOfAllEdges = []
    for row in read_tsv:
        if row[0] != row[1]:
            listOfAllEdges.append(row)
    
    return listOfAllEdges

def readGafFile(fileName):
    gaf_file = open(fileName)
    read_gaf = csv.reader(gaf_file, delimiter="\t")
    cnt = 1
    gafData = []
    for row in read_gaf:
        if cnt >= 41:#Data starts at row 47
            if row[8] == 'P':
                gafData.append(row) 
        cnt += 1
    return gafData

def generateRandomList(listOfAllEdges, numOfIterations, modelNeighborsLists):
    randomList = copy.deepcopy(listOfAllEdges)
    numOfEdges = len(listOfAllEdges)
    numOfSwitches = numOfEdges * numOfIterations
    cnt = 0
    while cnt < numOfSwitches:
        edge0 = random.randrange(0, numOfEdges, 1)
        edge1 = random.randrange(0, numOfEdges, 1)
        while edge0 == edge1:
            edge1 = random.randrange(0, numOfEdges, 1)
        node00 = randomList[edge0][0]
        node01 = randomList[edge0][1]
        node10 = randomList[edge1][0]
        node11 = randomList[edge1][1]
        if (node00 == node10) or (node00 == node11) or (node01 == node10) or (node01 == node11):
            continue
        if (node00 in modelNeighborsLists[node11]) or (node10 in modelNeighborsLists[node01]):
            continue
        
        tmp = randomList[edge0][1]
        randomList[edge0][1] = randomList[edge1][1]
        randomList[edge1][1] = tmp
        cnt += 1
    
    return randomList

def createGAndDics(listOfAllEdges):
    G = nx.Graph()
    modelNeighborsLists = {}
    modelNumOfNeighbors = {}
    for edge in listOfAllEdges:
        if modelNumOfNeighbors.get(edge[0]) is None:
            G.add_node(edge[0])
            modelNumOfNeighbors[edge[0]] = 1
            modelNeighborsLists[edge[0]] = [edge[1]]
        else: 
            modelNumOfNeighbors[edge[0]] += 1
            modelNeighborsLists[edge[0]] += [edge[1]]

        if modelNumOfNeighbors.get(edge[1]) is None:
            G.add_node(edge[1])
            modelNumOfNeighbors[edge[1]] = 1
            modelNeighborsLists[edge[1]] = [edge[0]]
        else: 
            modelNumOfNeighbors[edge[1]] += 1
            modelNeighborsLists[edge[1]] += [edge[0]]

        G.add_edge(edge[0], edge[1])
    return G, modelNeighborsLists, modelNumOfNeighbors

