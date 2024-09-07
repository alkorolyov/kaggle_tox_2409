import pandas as pd
import numpy as np
from copy import deepcopy
import networkx as nx
from networkx.algorithms.approximation import min_weighted_vertex_cover
from src.config import mem

def get_corr(df):
    return df.corr()


def _non_corr_features(corr, weights=None, threshold: float = 0.95):
    ids = (np.triu(corr, k=1) > threshold).nonzero()
    G = nx.Graph()

    if weights is None:
        G.add_nodes_from(np.arange(len(corr)))
    else:
        for i in range(len(corr)):
            G.add_node(i, weight=weights[i])

    for i in range(len(ids[0])):
        G.add_edge(ids[0][i], ids[1][i])

    nodes = deepcopy(G.nodes)

    for c in nx.connected_components(G):
        nodes -= min_weighted_vertex_cover(G.subgraph(c), weight='weight')

    non_corr_ids = np.asarray(list(nodes))

    return non_corr_ids


def non_corr_ids(corr: pd.DataFrame, threshold: float = 0.95):
    """
    Filters correlated features to only those that are not correlated above
    the indicated threshold. Uses graph min_weighted_vertex_cover to compute the
    maximum number of non-correlated features.
    :param corr: correlation matrix with last column as target
    :param threshold: correlation threshold
    :return: non-correlated features indexes
    """

    feat_corr = corr.iloc[:-1, :-1].abs()

    # to minimize weights in min vertex cover algorithm
    # we set weights as 1 - abs(corr with target)
    weights = 1 - corr.iloc[:, -1].abs()[:-1]

    ids = (np.triu(feat_corr, k=1) > threshold).nonzero()

    G = nx.Graph()

    for i in range(len(feat_corr)):
        G.add_node(i, weight=weights.iat[i])
    for i in range(len(ids[0])):
        G.add_edge(ids[0][i], ids[1][i])

    nodes = deepcopy(G.nodes)

    for c in nx.connected_components(G):
        nodes -= min_weighted_vertex_cover(G.subgraph(c), weight='weight')

    non_corr = np.asarray(list(nodes))

    return non_corr


def non_corr_features(X: pd.DataFrame, y: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    df = pd.concat([X, y], axis=1)
    corr = mem.cache(get_corr)(df)
    ids = non_corr_ids(corr, threshold)
    return X.iloc[:, ids]
