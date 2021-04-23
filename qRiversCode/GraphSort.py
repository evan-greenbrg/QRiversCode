import numpy as np
import pandas
import matplotlib.pyplot as plt
import networkx as nx
import itertools


def JoinComponents(H, G):
    G.remove_edges_from(list(G.edges))
    # Get number of components
    S = [H.subgraph(c).copy() for c in nx.connected_components(H)]
    while len(S) != 1:
        # Get node positions
        node_attributes = nx.get_node_attributes(H, 'pos')

        # Iterate through all pairs ofconnected components
        Scombs = list(itertools.product(S, S))
        for idx, pair in enumerate(Scombs):
            if pair[0] == pair[1]:
                continue
            # Get all combinations between two lists
            S0_nodes = list(pair[0].nodes)
            S1_nodes = list(pair[1].nodes)
            combs = list(itertools.product(S0_nodes, S1_nodes))

            # Get all lengths between each graph nodes
            lengths = []
            for comb in combs:
                pos1 = node_attributes[comb[0]]
                pos2 = node_attributes[comb[1]]
                lengths.append(np.linalg.norm(np.array(pos1) - np.array(pos2)))

            # Get shortest length index and use that edge
            i = np.argsort(lengths)[0]
            comp_edge = combs[i]
            length = lengths[i]

            G.add_edge(*comp_edge, length=length)

        # Iterate trhough components to find shortest component for each node
        for s in S:
            comp_nodes = list(s.nodes)
            min_edges = []
            min_lengths = []
            for node in comp_nodes:
                edges = list(G.edges(node))
                lengths = [G.get_edge_data(*e)['length'] for e in edges]
                if not lengths:
                    continue
                min_edges.append(edges[np.argmin(lengths)])
                min_lengths.append(lengths[np.argmin(lengths)])

            H.add_edge(
                *min_edges[np.argmin(min_lengths)], 
                length=np.min(min_lengths)
            )

        G.remove_edges_from(list(G.edges))
        # Get number of components
        S = [H.subgraph(c).copy() for c in nx.connected_components(H)]
        print(len(S))

    return H

def getGraph(path, xcol='Easting', ycol='Northing'):
    df = pandas.read_csv(path)
    start = 0
    end = len(df)-1
    data = np.array(df[[xcol, ycol]])
    tmp = [tuple(i) for i in data]

    G = nx.Graph()
    H = nx.Graph()
    for idx, row in enumerate(tmp):
        G.add_node(idx, pos=row)
        H.add_node(idx, pos=row)

    # Add all edges
    for idx, nodeA in enumerate(tmp):
        for jdx, nodeB in enumerate(tmp):
            if idx == jdx:
                continue
            else:
                length = np.linalg.norm(np.array(nodeA) - np.array(nodeB))
                G.add_edge(idx, jdx, length=length)

    # Reduce number of edges so each node only has two edges
    for node in G.nodes():
        # Get all edge lengths 
        edge_lengths = np.empty((len(G.edges(node)),))
        edges = np.array(list(G.edges(node)))
        for idx, edge in enumerate(edges):
            edge_lengths[idx] = G.get_edge_data(*edge)['length']

        # Only select the two smallest lengths
        if (node == start) or (node == end):
            ks = np.argpartition(edge_lengths, 2)[:1]
        else:
            ks = np.argsort(edge_lengths)[:2]

        use_edges = [tuple(i) for i in edges[ks]]

        # Add the filtered edges to the H network
        for edge in use_edges:
            length = G.get_edge_data(*edge)['length']
            H.add_edge(*edge, length=length)

    H = JoinComponents(H, G)

    return H


def saveGraph(G, opath):
    """Save the graph to the outpath
    """
    nx.write_gpickle(G, opath)


def GraphSort(H, df):
    df['idx'] = df.index
    start = 0
    end = list(H.nodes)[-1]

    # Sort the shuffled DataFrame
    path = np.array(
        nx.shortest_path(H, source=start, target=end, weight='length')
    )

    return df.iloc[path].reset_index(drop=True)
    centerlineSort2 = df.iloc[path].reset_index(drop=True)


if __name__=='__main__':
    plt.plot(ShuffDf['Easting'], ShuffDf['Northing'])
    plt.scatter(ShuffDf['Easting'], ShuffDf['Northing'])
    plt.plot(ReSortDf['Easting'], ReSortDf['Northing'])
    plt.scatter(ReSortDf['Easting'], ReSortDf['Northing'])
    plt.show()
