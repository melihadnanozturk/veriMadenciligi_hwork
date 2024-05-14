import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os

def find_optimal_clusters(df, threshold=0.1, max_clusters=None):

    inertia_list = []
    max_clusters = len(df.columns) if max_clusters is None else max_clusters

    for k in range(1, max_clusters + 1):
        kmeans_model = KMeans(n_clusters=k)
        kmeans_model.fit(df)
        inertia_list.append(kmeans_model.inertia_)

        if len(inertia_list) > 1:
            improvement = (inertia_list[-2] - inertia_list[-1]) / inertia_list[-2] * 100
            if improvement < threshold:
                return k, inertia_list

    return None, inertia_list


def squeeze_clusters(df, kmeans, target_clusters=None):

    while kmeans.n_clusters_ > target_clusters:
        # Calculate pairwise distances between cluster centroids
        distances = pd.DataFrame(kmeans.cluster_centers_).apply(
            lambda x: pd.Series(x).sub(kmeans.cluster_centers_).pow(2).sum(1), axis=1)

        # Find the indices of the two closest clusters (modify if using a different distance metric)
        min_idx1, min_idx2 = distances[distances != 0].unstack().idxmin()

        # Merge clusters
        df.loc[df['Cluster'] == min_idx2, 'Cluster'] = min_idx1

        # Update kmeans model
        kmeans.labels_ = df['Cluster'].to_numpy()
        kmeans.cluster_centers_ = df.groupby('Cluster').mean().to_numpy()

    return kmeans

os.chdir('C:/Users/Adnan/Desktop/Ders Notları/3-2.D/Veri Madenciliği/VM Ödev')
data = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')

# Handle blood pressure column
data[['P_High', 'P_Low']] = data['Blood Pressure'].str.split('/', expand=True)
data.drop(columns=["Blood Pressure"], inplace=True)

# Create dummy variables for categorical features
data = pd.get_dummies(data, columns=['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder'], dtype=int)

# Find optimal number of clusters
k, inertia = find_optimal_clusters(data)

if k is not None:
    print(f"Estimated optimal number of clusters: {k}")

    # Scale Data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Initial clustering
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data_scaled)
    data['Cluster'] = kmeans.labels_

    # Apply Squeezer algorithm
    kmeans = squeeze_clusters(data.copy(), kmeans, target_clusters=3)  # Example

    # Calculate Silhouette Score
    if len(set(kmeans.labels_)) > 1:
        print(f"Average Silhouette Score: {silhouette_score(data_scaled, kmeans.labels_)}")
    else:
        print("Not enough clusters to calculate silhouette score.")

    # Visualization
    features = ['Age', 'Quality of Sleep']
    sns.scatterplot(
        x=features[0],
        y=features[1],
        hue="Cluster",
        palette="deep",
        data=data,
    )

    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.title('Distribution of Clusters')
    plt.legend(title='Cluster')
    plt.show()