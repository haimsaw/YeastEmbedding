import graphUtils
import networkx as nx
import clusterUtils
import numpy as np
import matplotlib.pyplot as plt


def main():
    gafData = graphUtils.readGafFile("./db/goa_human.gaf")
    listOfAllEdges = graphUtils.generateEdgesListFromFile("./db/huri_symbol.tsv")
    G, modelNeighborsLists, modelNumOfNeighbors = graphUtils.createGAndDics(listOfAllEdges)

    #Retrieve the 3-core of G
    print('Retreiving 3-core of G')
    G3Core = nx.algorithms.core.k_core(G, 3)

    #Creating the adjancey matrix in scipy sparse matrix format, as recommended
    print('Reformatting G to a sparse format')
    adjMatrix = nx.to_scipy_sparse_matrix(G3Core)

    #Run MCL algorithm
    infParam = 2
    expParam = 2
    clusters = clusterUtils.mclAlg(adjMatrix.copy(), infParam, expParam)

    #Computing number of each annotation in G
    numOfAnnotationsInG = clusterUtils.getNumOfAnnotationsInG(G3Core, gafData)

    #Computing the functional enrichment of each of the clusters
    print('Computing functional enrichment of clusters')
    clustersAnnotation, clustersPVal = clusterUtils.computeClustersFuncEnrichment(G3Core, clusters, gafData, numOfAnnotationsInG)

    print(f'score={clusterUtils.partition_score(G, clusters, gafData)}')
    #Create txt files
    print('Writing to files')
    clusterUtils.createTxtFiles(G3Core, clusters, clustersPVal, clustersAnnotation, 'clustered_proteins.txt', 'clusters_P.txt')


if __name__ == "__main__":
    main()