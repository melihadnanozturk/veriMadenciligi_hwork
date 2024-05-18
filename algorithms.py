from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes
from sklearn.cluster import KMeans


def k_modes(data, clusterNumber):
    kmodes = KModes(n_clusters=clusterNumber, init='Cao', n_init=20, verbose=0)
    clusters = kmodes.fit_predict(data)
    return clusters


def k_means(data, clusterNumber):
    kmeans = KMeans(n_clusters=clusterNumber)
    clusters = kmeans.fit_predict(data)
    return clusters


def k_prototypes(data, dataCategorical, clusterNumber):
    kproto = KPrototypes(n_clusters=clusterNumber, init='Huang', n_init=10, verbose=0)
    clusters = kproto.fit_predict(data, categorical=dataCategorical)
    return clusters
