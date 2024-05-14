import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
from sklearn.cluster import KMeans

sns.set()


## Veriler Cekilir
os.chdir('C:/Users/Adnan/Desktop/Ders Notları/Veri Madenciliği/VM Ödev')
dataset = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')
df = pd.DataFrame(dataset)

df['Gender'] = df['Gender'].map(({'Male': 0, 'Female': 1}))
jobs = df['Occupation'].unique()
bmiCategories = df['BMI Category'].unique()
df[['Pressure_High', 'Pressure_Low']] = df['Blood Pressure'].str.split('/', expand=True)
df.drop(columns=["Blood Pressure"], inplace=True)
df.fillna({'Sleep Disorder': "Any", }, inplace=True)
disorders = df['Sleep Disorder'].unique()

for i, job in enumerate(jobs):
    df.loc[df['Occupation'] == job, 'Occupation'] = i

for i, bmi in enumerate(bmiCategories):
    df.loc[df['BMI Category'] == bmi, 'BMI Category'] = i

for i, disorder in enumerate(disorders):
    df.loc[df['Sleep Disorder'] == disorder, 'Sleep Disorder'] = i

print(df)
# plt.scatter(dataset["Stress Level"], dataset["Physical Activity Level"])
# plt.ylabel("Stress Level")
# plt.xlabel("Physical Activity Level")
# plt.show()

x = df.copy()
kmeans = KMeans(2)
kmeans.fit(x)  # Model oluşturuldu

# clusters = x.copy()
# clusters['cluster_pred'] = kmeans.fit_predict(x)
#
# print(clusters)
#
# plt.scatter(clusters["Stress Level"], clusters["Physical Activity Level"], c=clusters['cluster_pred'], cmap='rainbow')
# plt.xlabel("Sleep Duration")
# plt.xlabel("Physical Activity Level")
# plt.show()

centers = kmeans.cluster_centers_
labels = kmeans.labels_
df["clusters"] = labels
print(df["Sleep Duration"])

plt.scatter(df['Stress Level'], df['Sleep Duration'], c=df['clusters'], cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x')
plt.xlabel('Sleep Duration')
plt.ylabel('Stress Level')
plt.title('K-Means ile Oluşturulan Kümeler')
plt.show()
