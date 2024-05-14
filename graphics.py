import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
sns.set()
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes



def createSutunGraphics(data, column, title):
    tekrarlar = data[column].value_counts()

    # Sütun grafiği için etiketler ve verileri ayıralım
    etiketler = tekrarlar.index
    veriler = tekrarlar.values

    # Sütun grafiğini çizelim
    plt.figure(figsize=(10, 6))
    plt.bar(etiketler, veriler, color='skyblue')
    plt.xlabel(column)
    plt.ylabel('Tekrar Sayısı')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()

def createPieChart(data, column):
    tekrarlar = data[column].value_counts()

    # Pasta grafiği için etiketler ve verileri ayıralım
    etiketler = tekrarlar.index
    veriler = tekrarlar.values

    # Pasta grafiğini çizelim
    plt.figure(figsize=(8, 6))
    plt.pie(veriler, labels=etiketler, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')  # Dairesel şeklin korunması için
    plt.show()

def create_cluster_graphics(pca_result,clusters, title):
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=clusters, cmap='viridis')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(title)
    plt.colorbar(label='Cluster')
    plt.show()

def create_elbow_graphics_for_kmeans(data):
    cluster_numbers = range(1, 11)
    inertia_values = []

    # Farklı küme sayıları için kmeans algoritmasını çalıştırın ve küme içi varyansı hesaplayın
    for k in cluster_numbers:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertia_values.append(kmeans.inertia_)

    # Elbow grafiğini çizin
    plt.plot(cluster_numbers, inertia_values, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for KMeans')
    plt.show()

def create_elbow_graphics_for_kmmodes(data):
    cluster_numbers = range(1, 11)
    cost_values = []

    # Farklı küme sayıları için kmodes algoritmasını çalıştırın ve küme içi maliyeti hesaplayın
    for k in cluster_numbers:
        kmodes = KModes(n_clusters=k, init='Huang', n_init=5, verbose=0)
        kmodes.fit_predict(data)
        cost_values.append(kmodes.cost_)

    # Elbow grafiğini çizin
    plt.plot(cluster_numbers, cost_values, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Cost')
    plt.title('Elbow Method for KModes')
    plt.show()

def create_elbow_graphics_for_kprototypes(data,categorical):
    cluster_numbers = range(1, 11)
    cost_values = []

    # Farklı küme sayıları için kprototypes algoritmasını çalıştırın ve küme içi maliyeti hesaplayın
    for k in cluster_numbers:
        kproto = KPrototypes(n_clusters=k, init='Cao')
        clusters = kproto.fit_predict(data, categorical=categorical)  # Kategorik sütunlar için indeksleri belirtin
        cost_values.append(kproto.cost_)

    # Elbow grafiğini çizin
    plt.plot(cluster_numbers, cost_values, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Cost')
    plt.title('Elbow Method for KPrototypes')
    plt.show()

