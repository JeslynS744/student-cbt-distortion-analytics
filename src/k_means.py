import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_data_path = os.path.join(BASE_DIR, 'data', 'processed', 'original_preprocessed_student_mental_health_data.csv')
output_dir = os.path.join(BASE_DIR, 'results', 'plots')
output_file_path = os.path.join(BASE_DIR, 'data', 'processed', 'original_clustered_student_mental_health_data.csv')

os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(input_data_path)

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].str.strip()
    df.replace("", np.nan, inplace=True)

df.dropna(subset=["age", "gender", "Course"], inplace=True)

gender_map = {0: "Female", 1: "Male"}
df["gender"] = df["gender"].map(gender_map)
print(df['gender'].value_counts())

if 'aware_of_cbt' in df.columns:    
    awareness_map = {0: "No", 2: "Yes"}
    df['aware_of_cbt'] = df['aware_of_cbt'].map(awareness_map)
    print(df['aware_of_cbt'].value_counts())
else:
    print("Column 'aware_of_cbt' not found in the dataframe.")

df['age'] = pd.to_numeric(df['age'], errors='coerce')
df.dropna(subset=['age'], inplace=True)
df['age'] = df['age'].astype(int)
df["Age_Group"] = pd.cut(df["age"], bins=[17, 20, 23, 25], labels=["18-21", "22-24", "24+"])

cbt_columns = {
    "Black & White Thinking": [5, 6, 7],
    "Overgeneralization": [8, 9, 10],
    "Personalization": [11, 12, 13],
    "Catastrophizing": [14, 15, 16],
    "Should Statements": [17, 18, 19]
}

anxiety_cols = [20, 21, 22]
depression_cols = [23, 24, 25]
THRESHOLD = 2.5

def convert_to_numeric(df, columns):
    for col in columns:
        if col < df.shape[1]:
            df.iloc[:, col] = pd.to_numeric(df.iloc[:, col], errors='coerce')
    return df

df = convert_to_numeric(df, 
    [col for sublist in list(cbt_columns.values()) + list(anxiety_cols) + list(depression_cols) for col in (sublist if isinstance(sublist, list) else [sublist])]
)

for key, cols in cbt_columns.items():
    valid_cols = [col for col in cols if col < df.shape[1]]
    if valid_cols:
        df[key] = df.iloc[:, valid_cols].mean(axis=1)
    else:
        df[key] = np.nan

course_map = {
    "bsc": "BSc", "msc": "MSc", "be": "BE", "btech": "BTech",
    "bba": "BBA", "bca": "BCA", "bcom": "BCOM", "mbbs": "MBBS"
}
df["Course_Grouped"] = df["Course"].str.lower().map(course_map).fillna("Other")
df["Anxiety"] = df.iloc[:, anxiety_cols].mean(axis=1)
df["Depression"] = df.iloc[:, depression_cols].mean(axis=1)

for key in cbt_columns.keys():
    df[f"{key}_Recognized"] = df[key] > THRESHOLD

df["Anxiety_Recognized"] = df["Anxiety"] > THRESHOLD
df["Depression_Recognized"] = df["Depression"] > THRESHOLD

cluster_features = [
    "Black & White Thinking", "Overgeneralization", "Personalization",
    "Catastrophizing", "Should Statements", "Anxiety", "Depression"
]
df_cluster = df[cluster_features].copy()
df_cluster.dropna(inplace=True)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_cluster)
scaled_df = pd.DataFrame(scaled_features, columns=cluster_features, index=df_cluster.index)

silhouette_scores = []
range_n_clusters = range(2, 11)

for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_df)
    silhouette_avg = silhouette_score(scaled_df, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f"For n_clusters = {n_clusters}, the average silhouette_score is : {silhouette_avg}")

plt.figure(figsize=(10, 6))
plt.plot(range_n_clusters, silhouette_scores, marker='o')
plt.xlabel("Number of clusters")
plt.ylabel("Silhouette score")
plt.title("Silhouette Score vs. Number of Clusters")
plt.xticks(range_n_clusters)
plt.grid(True)
plt.savefig(os.path.join(output_dir, "silhouette_score.png"), bbox_inches='tight')
plt.close()

n_clusters = np.argmax(silhouette_scores) + 2
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df_cluster['Cluster'] = kmeans.fit_predict(scaled_df)

df['Cluster'] = pd.Series(np.nan, index=df.index)
df.loc[df_cluster.index, 'Cluster'] = df_cluster['Cluster']

cluster_summary = df.groupby('Cluster')[cluster_features].mean()
cluster_summary = cluster_summary.apply(pd.to_numeric, errors='coerce')
cluster_summary.fillna(0, inplace=True)

plt.figure(figsize=(12, 8))
sns.heatmap(cluster_summary, annot=True, cmap="YlGnBu")
plt.title("Cluster Profiles: Average CBT Distortions, Anxiety, and Depression")
plt.savefig(os.path.join(output_dir, "cluster_profiles.png"), bbox_inches='tight')
plt.close()

print("\nCluster Summary:")
print(cluster_summary)

pair_plot_features = ["Black & White Thinking", "Anxiety", "Depression", "Cluster"]
pp = sns.pairplot(df[pair_plot_features], hue="Cluster", palette="viridis")
plt.suptitle("Pair Plot of Clusters with Select Features", y=1.02)
plt.savefig(os.path.join(output_dir, "cluster_pair_plot.png"), bbox_inches='tight')
plt.close('all')

print("\nDataFrame with Cluster Assignments:")
print(df[['age', 'gender', 'Course_Grouped', 'Cluster']].head())

awareness_map_numeric = {"No": 0, "Yes": 2}
df['aware_of_cbt'] = df['aware_of_cbt'].map(awareness_map_numeric)

df.to_csv(output_file_path, index=False)
print(f"Clustered data saved to: {output_file_path}")

def plot_cbt_distributions(df):
    recognized_counts = {dist: df[f"{dist}_Recognized"].sum() for dist in cbt_columns.keys()}
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(recognized_counts.keys()), y=list(recognized_counts.values()), palette="muted")
    plt.xlabel("CBT Distortions")
    plt.ylabel("No. of Recognized Cases")
    plt.title("Recognized CBT Distortions Among Individuals")
    total_people = len(df)
    total_affected = sum(recognized_counts.values())

    for i, v in enumerate(recognized_counts.values()):
        plt.text(i, v + 1, v, ha='center', fontsize=12)

    plt.figtext(0.85, 0.9, f"Total People: {total_people}\nTotal Affected: {total_affected}", fontsize=12, ha="right")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cbt_distributions.png"), bbox_inches='tight')
    plt.close()

def plot_anxiety_depression(df):
    counts = {
        "Anxiety": df["Anxiety_Recognized"].sum(),
        "Depression": df["Depression_Recognized"].sum()
    }
    plt.figure(figsize=(6, 4))
    sns.barplot(x=list(counts.keys()), y=list(counts.values()), palette="coolwarm")
    plt.xlabel("Condition")
    plt.ylabel("No. of Recognized Cases")
    plt.title("Recognized Anxiety & Depression Cases")

    for i, v in enumerate(counts.values()):
        plt.text(i, v + 1, v, ha='center', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "anxiety_depression.png"), bbox_inches='tight')
    plt.close()

def plot_gender_cbt(df):
    gender_cbt = df.groupby("gender")[[f"{dist}_Recognized" for dist in cbt_columns.keys()]].sum()
    if gender_cbt.empty:
        print("No data to plot for Gender-wise CBT Distortions.")
        return

    gender_cbt.T.plot(kind='bar', figsize=(10, 6), colormap='viridis')
    plt.xlabel("CBT Distortions")
    plt.ylabel("No. of Recognized Cases")
    plt.title("Gender-wise CBT Distortions")
    plt.legend(title="Gender")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gender_cbt.png"), bbox_inches='tight')
    plt.close()

def plot_course_distribution(df):
    course_counts = df["Course_Grouped"].value_counts()
    plt.figure(figsize=(8, 5))
    sns.barplot(x=course_counts.index, y=course_counts.values, palette="cubehelix")
    plt.xlabel("Course")
    plt.ylabel("No. of Respondents")
    plt.title("Course-wise Distribution of Responses")

    for i, v in course_counts.items():
        plt.text(i, v + 1, v, ha='center', fontsize=12)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "course_distribution.png"), bbox_inches='tight')
    plt.close()

def plot_age_cbt(df):
    age_cbt = df.groupby("Age_Group")[[f"{dist}_Recognized" for dist in cbt_columns.keys()]].sum()
    if age_cbt.empty:
        print("No data to plot for Age-wise CBT Distortions.")
        return

    age_cbt.T.plot(kind='bar', figsize=(10, 6), colormap='plasma')
    plt.xlabel("CBT Distortions")
    plt.ylabel("No. of Recognized Cases")
    plt.title("Age-wise CBT Distortions Distribution")
    plt.legend(title="Age Group")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "age_cbt.png"), bbox_inches='tight')
    plt.close()

plot_cbt_distributions(df)
plot_anxiety_depression(df)
plot_gender_cbt(df)
plot_course_distribution(df)
plot_age_cbt(df)
print("K-Means analysis and visual plots generated successfully!")