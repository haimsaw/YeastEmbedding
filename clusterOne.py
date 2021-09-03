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
    # G3Core = nx.algorithms.core.k_core(G, 3)
    G3Core = G

    clusters = clusterUtils.extractClusteOneClusters("./output/clusterOneOut.txt", list(G3Core.nodes))


    #Computing number of each annotation in G
    numOfAnnotationsInG = clusterUtils.getNumOfAnnotationsInG(G3Core, gafData)

    #Computing the functional enrichment of each of the clusters
    print('Computing functional enrichment of clusters')
    clustersAnnotation, clustersPVal = clusterUtils.computeClustersFuncEnrichment(G3Core, clusters, gafData, numOfAnnotationsInG)

    print(f'score={clusterUtils.partition_score(G, clusters, gafData)}')
    #Create txt files
    print('Writing to files')
    clusterUtils.createTxtFiles(G3Core, clusters,  gafData, clustersPVal, clustersAnnotation, 'clustered_proteins_clusterOne.txt', 'clusters_P_clusterOne.txt', 'summary_clusterOne.txt')

    # for i, cluster in enumerate(clusters):
    #     if len(cluster > 10):
    #         nodes = np.array(G.nodes)[list(cluster)]
    #         H = G.subgraph(nodes)
    #         print(f'\n\n nodes={len(nodes)} ')
    #         nx.draw(H, node_size=70)
    #         plt.show()

if __name__ == "__main__":
    main()