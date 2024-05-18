from sklearn.decomposition import PCA
import algorithms
import datas
import graphics
from sklearn import preprocessing
from sklearn.metrics import silhouette_score

pca = PCA(n_components=2)

path = 'C:/Users/Adnan/Desktop/Ders Notları/3-2.D/Veri Madenciliği/VM Ödev';
csvName = 'Sleep_health_and_lifestyle_dataset.csv';

# DATALARI GETIRIYORUZ
df = datas.get_data(path, csvName);

# NULL KONTROLU
# print(df.isnull())

# COLON SUTUN GRAFIGI OLUSTURUYORUZ
graphics.createSutunGraphics(df, 'Quality of Sleep', 'Quality Of Sleep Dağılımı');

# VERILERI BELLI ORANDA SILIYORUZ / KOLONUN ORTALAMASINI VERIYORUZ
datas.remove_data_by_rate_and_fill_column_mean(0.05, "Sleep Duration", df);

# COLON SUTUN GRAFIGI OLUSTURUYORUZ
graphics.createSutunGraphics(df, 'BMI Category', 'BMI Category Dağılımı');

# VERILERI BELLI ORANDA SILIYORUZ / ISTEDIGIMIZ BIR DEGERI VERIYORUZ
datas.remove_data_by_rate_and_fill_column_values(0.05, 'BMI Category', df, "NOT_KNOW")
print(df['BMI Category'])
graphics.createSutunGraphics(df, 'BMI Category', 'BMI Category Dağılımı');

# PIE CHART CIZIYORUZ
graphics.createPieChart(df, 'Sleep Disorder')
df["Sleep Disorder"] = df["Sleep Disorder"].fillna(value="HEALTHY")
graphics.createPieChart(df, 'Sleep Disorder')

# KAN BASINCI COLUMN'U MANTIKSAL OLARAK AYIRIYORUZ
df[['P_High', 'P_Low']] = df['Blood Pressure'].str.split('/', expand=True)
df.drop(columns=["Blood Pressure"], inplace=True)

# ILGILI KOLONLARI ENCODE EDIYORUZ
df = datas.encode_data_by_labelEncoded(df)

# KOLONLARIN ISIMLERINI DEGISTIRIYORUZ
df = datas.rename_columns(df)
df = df.drop(columns=["Person ID"])

# ILGILI KOLONLAR ICIN NORMALIZASYON
df[['2', '6', '9', '10', '12', '13']] = preprocessing.minmax_scale(df[['2', '6', '9', '10', '12', '13']])

# KATEGORI ICEREN BILGILERI AYIRIYORUZ
df_wout_categories = df.drop(columns=["1", "3", "8", "11"])

# ELBOW GRAFIKLERINI OLUSTURUYORUZ
graphics.create_elbow_graphics_for_kmeans(df)
graphics.create_elbow_graphics_for_kmmodes(df)
graphics.create_elbow_graphics_for_kprototypes(df, categorical=[1, 3, 8, 11])

# print(df)

##   -----------************----------   KMODES ALGORITMASI  -----------************----------
data_for_kmodes = df.copy()
clusters_kmodes = algorithms.k_modes(data=data_for_kmodes, clusterNumber=5)
pca_kmodes = pca.fit_transform(data_for_kmodes)
# kmodes pca ile grafik haline getirme / performans ölçümü
graphics.create_cluster_graphics(pca_result=pca_kmodes, clusters=clusters_kmodes, title="KMODES CLUSTERS")
silhouette_score_kmodes = silhouette_score(data_for_kmodes, clusters_kmodes)
print("silhouette_score_kmodes : ", silhouette_score_kmodes)

##   -----------************----------   KMEANS ALGORITMASI  -----------************----------
data_for_kmeans = df.copy()
clusters_kmeans = algorithms.k_means(data=data_for_kmeans, clusterNumber=5)
pca_kmeans = pca.fit_transform(data_for_kmeans)
graphics.create_cluster_graphics(pca_result=pca_kmeans, clusters=clusters_kmeans, title="KMEANS CLUSTERS")
silhouette_score_kmeans = silhouette_score(data_for_kmeans, clusters_kmeans)
print("silhouette_score_kmeans : ", silhouette_score_kmeans)

##   -----------************----------   KPROTOTYPES ALGORITMASI  -----------************----------
data_for_kprototpyes = df.copy()
categorical = [1, 3, 8, 11]
clusters_kprototypes = algorithms.k_prototypes(data=data_for_kprototpyes, dataCategorical=categorical, clusterNumber=5)
pca_kprototypes = pca.fit_transform(data_for_kprototpyes)
graphics.create_cluster_graphics(pca_result=pca_kprototypes, clusters=clusters_kprototypes,
                                 title="KPROTOTYPES CLUSTERS")
silhouette_score_kprototpyes = silhouette_score(data_for_kprototpyes, clusters_kprototypes)
print("silhouette_score_kprototpyes : ", silhouette_score_kprototpyes)

##   -----------************----------   WITHOUT CATEGORY  -----------************----------

##   -----------************----------   KMODES ALGORITMASI  -----------************----------
data_for_kmodes_wout = df_wout_categories.copy()
clusters_kmodes_wout = algorithms.k_modes(data=data_for_kmodes_wout, clusterNumber=5)
pca_kmodes_wout = pca.fit_transform(data_for_kmodes_wout)
# kmodes pca ile grafik haline getirme / performans ölçümü
graphics.create_cluster_graphics(pca_result=pca_kmodes_wout, clusters=clusters_kmodes_wout,
                                 title="KMODES WOUT CATEGORY CLUSTERS")
silhouette_score_kmodes_wout = silhouette_score(data_for_kmodes_wout, clusters_kmodes_wout)
print("silhouette_score_kmodes_wout : ", silhouette_score_kmodes_wout)

##   -----------************----------   KMEANS ALGORITMASI  -----------************----------
data_for_kmeans_wout = df_wout_categories.copy()
clusters_kmeans_wout = algorithms.k_means(data=data_for_kmeans_wout, clusterNumber=5)
pca_kmeans_wout = pca.fit_transform(data_for_kmeans_wout)
graphics.create_cluster_graphics(pca_result=pca_kmeans_wout, clusters=clusters_kmeans_wout,
                                 title="KMEANS WOUT CATEGORY CLUSTERS")
silhouette_score_kmeans_wout = silhouette_score(data_for_kmeans_wout, clusters_kmeans_wout)
print("silhouette_score_kmeans_wout : ", silhouette_score_kmeans_wout)

# data_for_kprototpyes_wout = df_wout_categories.copy();
# categorical = [1, 3, 8, 11]
# clusters_kprototypes_wout = algorithms.k_prototypes(data=data_for_kprototpyes_wout, dataCategorical=None, clusterNumber=5)
# pca_kprototypes_wout = pca.fit_transform(clusters_kprototypes_wout)
# graphics.create_cluster_graphics(pca_result=pca_kprototypes_wout,clusters=clusters_kprototypes_wout, title="KPROTOTYPES WOUT CATEGORY CLUSTERS")
