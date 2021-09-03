import sklearn.preprocessing
from scipy.sparse import isspmatrix, dok_matrix, csc_matrix
from scipy.stats import hypergeom
import numpy as np
from itertools import starmap
from statistics import mean
import csv
import random

def extractClusteOneClusters(filename, nodesList):
    tsv_file = open(filename)
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    clusters = []
    for row in read_tsv:
        cluster = []
        for protein in row:
            cluster.append(nodesList.index(protein))
        clusters.append(tuple(cluster))
    return clusters

def mclAlg(adjMatrix, infParam, expParam):
    print('Running MCL with inflation param: ' + str(round(infParam, 2)) + ', expansion param: ' + str(expParam))
    maxIterations = 20
    pruneThresh = 1e-6
    convergeThresh = 1e-8

    #Add self loops
    shape = adjMatrix.shape
    adjMatrixDok = adjMatrix.todok()
    for i in range(shape[0]):
        adjMatrixDok[i, i] = 1
    adjMatrix = adjMatrixDok.tocsc()

    #Normalize
    adjMatrix = sklearn.preprocessing.normalize(adjMatrix, norm="l1", axis=0)

    #Start iterations
    for i in range(maxIterations):
        print('Iteration number: ' + str(i + 1))
        prevAdjMatrix = adjMatrix.copy()

        #Expansion
        adjMatrix = adjMatrix ** expParam

        #Inflation
        adjMatrix = adjMatrix.power(infParam)
        #Normalize
        adjMatrix = sklearn.preprocessing.normalize(adjMatrix, norm="l1", axis=0)

        #Prune
        pruneAdjMatrix = dok_matrix(adjMatrix.shape)
        pruneAdjMatrix[adjMatrix >= pruneThresh] = adjMatrix[adjMatrix >= pruneThresh]
        pruneAdjMatrix = pruneAdjMatrix.tocsc()
        #Keeping largest element in each column
        rowIdxs = adjMatrix.argmax(axis=0).reshape((adjMatrix.shape[1],))
        colIdxs = np.arange(adjMatrix.shape[1])
        pruneAdjMatrix[rowIdxs, colIdxs] = adjMatrix[rowIdxs, colIdxs]
        adjMatrix = pruneAdjMatrix

        #Check convergence
        convergeVal = np.abs(adjMatrix - prevAdjMatrix)
        if convergeVal.max() <= convergeThresh:
            print('Converged! converge val is: ' + str(convergeVal.max()))
            break

    print('Done. Retreiving clusters...')
    #Find attaractors
    attractors = adjMatrix.diagonal().nonzero()[0]

    #Get clusters
    clusters = set()
    for attractor in attractors:
        cluster = tuple(adjMatrix.getrow(attractor).nonzero()[1].tolist())
        if len(cluster) < 5:
            continue
        clusters.add(cluster)
    
    return sorted(list(clusters), key=len)


def getProteinAnnotation(node, gaf_data):
    return gaf_data.get(node, -1)


def getClusterAnnotation(cluster, nodesList, gaf_data):
    annotation = -1
    clusterAnnotations = {}
    maxAnnotation = ''
    maxNumber = 0
    for nodeIdx in cluster:
        nodeAnnotation = getProteinAnnotation(nodesList[nodeIdx], gaf_data)
        if nodeAnnotation == -1:
            continue
        if clusterAnnotations.get(nodeAnnotation) is None:
            clusterAnnotations[nodeAnnotation] = 1
        else:
            clusterAnnotations[nodeAnnotation] += 1
        
        if clusterAnnotations[nodeAnnotation] > maxNumber:
            maxNumber = clusterAnnotations[nodeAnnotation]
            maxAnnotation = nodeAnnotation
    
    if maxNumber > 2:
        annotation = maxAnnotation

    return annotation, maxNumber

def computeClustersFuncEnrichment(G, clusters, gafData, numOfAnnotationsInG):
    clustersPVal = []
    clustersAnnotation = [] 
    nodesList = list(G.nodes)
    numOfNodes = len(nodesList)

    for cluster in clusters:
        clusterAnnotation, numOfAnnotations = getClusterAnnotation(cluster, nodesList, gafData)
        clustersAnnotation.append(clusterAnnotation)
        if clusterAnnotation == -1:#Less than 3 protien
            clustersPVal.append(-1)
            continue
        #Computing p-val
        C = len(cluster)
        dist = hypergeom(numOfNodes, numOfAnnotationsInG[clusterAnnotation], C)
        p_val = dist.pmf(numOfAnnotations)
        p_val = -1 * np.log(p_val)
        clustersPVal.append(p_val)

    return clustersAnnotation, clustersPVal

def createTxtFiles(G, clusters, gafData, clustersPVal, clustersAnnotation, fileName1, fileName2, fileName3):
    nodesList = list(G.nodes)
    #First file
    with open('./output/' + fileName1, 'w') as f:
        f.write('protein name' + "\t" + 'cluster' + "\n")
        clusterCnt = 0
        for cluster in clusters:
            clusterCnt += 1
            for nodeIdx in cluster:
                f.write(nodesList[nodeIdx] + "\t" + str(clusterCnt) + "\n")
        f.close()
    #Second file
    with open('./output/' + fileName2, 'w') as f:
        f.write('cluster' + "\t" + 'p-value' + "\t" + 'GO-term' + "\n")
        clusterCnt = 0
        for cluster in clusters:
            clusterCnt += 1
            if clustersAnnotation[clusterCnt - 1] == -1:
                continue
            f.write(str(clusterCnt) + "\t" + str(round(clustersPVal[clusterCnt - 1],2)) + "\t" + str(clustersAnnotation[clusterCnt - 1]) + "\n")
        f.close()
    partitionScore, meanSize, maxSize, minSize = results(G, clusters, gafData)
    with open('./output/' + fileName3, 'w') as f:
        f.write('cluster num of clusters : ' + str(len(clusters)) + "\n")
        f.write('cluster partitionScore : ' + str(partitionScore) + "\n")
        f.write('cluster meanSize : ' + str(meanSize) + "\n")
        f.write('cluster maxSize : ' + str(maxSize) + "\n")
        f.write('cluster minSize: ' + str(minSize) + "\n")
        f.close()


def getNumOfAnnotationsInG(G, gafData):
    nodesList = list(G.nodes)
    numOfAnnotationsInG = {}
    for node in nodesList:
        nodeAnnotation = getProteinAnnotation(node, gafData)
        if nodeAnnotation == -1:
            continue
        if numOfAnnotationsInG.get(nodeAnnotation) is None:
            numOfAnnotationsInG[nodeAnnotation] = 1
        else:
            numOfAnnotationsInG[nodeAnnotation] += 1
    return numOfAnnotationsInG


def partition_score(G, clusters, gaf_data):
    num_of_annotations_in_g = getNumOfAnnotationsInG(G, gaf_data)
    clusters_annotation, clusters_P_Val = computeClustersFuncEnrichment(G, clusters, gaf_data, num_of_annotations_in_g)
    return sum(starmap(lambda cluster, pval: len(cluster) * max(pval, 0), zip(clusters, clusters_P_Val))) / len(G.nodes)

def clustersParams(clusters):
    lenList = []
    for cluster in clusters:
        lenList.append(len(cluster))
    return np.mean(np.array(lenList)), np.max(np.array(lenList)), np.min(np.array(lenList))

def results(G, clusters, gaf_data):
    partitionScore = partition_score(G, clusters, gaf_data)
    meanSize, maxSize, minSize = clustersParams(clusters)
    return partitionScore, meanSize, maxSize, minSize