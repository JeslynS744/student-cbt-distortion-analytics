import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_data_path = os.path.join(BASE_DIR, 'data', 'processed', 'original_clustered_student_mental_health_data.csv')
output_dir = os.path.join(BASE_DIR, 'results', 'plots')
conclusion_path = os.path.join(BASE_DIR, 'results', 'final_conclusion.txt')

os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.dirname(conclusion_path), exist_ok=True)

df = pd.read_csv(input_data_path)
exclude_cols = ['Cluster', 'timestamp', 'Email address', 'cbt_description',
                'anxiety_depression_description', 'College', 'Course',
                'Age_Group', 'Course_Grouped'] + [chr(i) for i in range(ord('A'), ord('U') + 1)]

X = df.drop(exclude_cols, axis=1, errors='ignore')
y = df['Cluster']
X = X.dropna(axis=1, how='all')
X = pd.get_dummies(X, drop_first=True)
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=imputer.feature_names_in_)
print(f"Missing values after imputation: {X.isna().sum().sum()}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBClassifier(objective='multi:softmax', num_class=len(np.unique(y)), n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    print(f"\n{name} Training:")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    results[name] = {'accuracy': accuracy, 'report': report, 'confusion_matrix': cm}
    print(f"Accuracy = {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(f"{name}: Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name.replace(' ', '_')}_confusion_matrix.png"))
    plt.close()

model_names = list(results.keys())
accuracy_scores = [results[name]['accuracy'] for name in model_names]
plt.figure(figsize=(10, 6))
plt.bar(model_names, accuracy_scores, color='skyblue')
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Model Comparison: Accuracy")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "model_comparison_accuracy.png"))
plt.close()

print("\nResults Summary:")
for name, result in results.items():
    print(f"\n{name}:")
    print(f"Accuracy = {result['accuracy']:.4f}")
    print("Confusion Matrix:")
    print(result['confusion_matrix'])

distortion_cols = [
    'Black & White Thinking', 'Overgeneralization', 'Personalization',
    'Catastrophizing', 'Should Statements'
]
avg_distortion_scores = df[distortion_cols].mean().sort_values(ascending=False)
most_observed_distortion = avg_distortion_scores.idxmax()
highest_score = avg_distortion_scores.max()

print("\n--- CBT Distortion Analysis ---")
print("Average scores for each distortion:")
print(avg_distortion_scores)
print(f"\nMost observed distortion: {most_observed_distortion} (avg score: {highest_score:.2f})")

df['Primary_Distortion'] = df[distortion_cols].idxmax(axis=1)
cluster_summary = df.groupby('Cluster').agg({
    'Anxiety': 'mean',
    'Depression': 'mean',
    'Primary_Distortion': lambda x: x.mode()[0] if not x.mode().empty else "None"
}).reset_index()

print("\n--- Cluster Summary ---")
print(cluster_summary)

with open(conclusion_path, "w") as f:
    f.write(avg_distortion_scores.to_string())
    f.write(f"\n\nMost observed distortion: {most_observed_distortion} (avg score: {highest_score:.2f})\n")
    f.write(cluster_summary.to_string(index=False))

top_cluster_info = cluster_summary[cluster_summary['Primary_Distortion'] == most_observed_distortion]
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model_accuracy = results[best_model_name]['accuracy']

print("\nBest Performing Model:", best_model_name, f"with Accuracy = {best_model_accuracy:.2%}")
print("\n--- Recommendations Based on Analysis ---")
print(f"1. The most frequently observed CBT distortion is: {most_observed_distortion} (Avg Score: {highest_score:.2f})")
if not top_cluster_info.empty:
    top_cluster_id = top_cluster_info['Cluster'].values[0]
    avg_anxiety = top_cluster_info['Anxiety'].values[0]
    avg_depression = top_cluster_info['Depression'].values[0]
    print(f"2. Cluster {int(top_cluster_id)} most commonly exhibits this distortion, with average anxiety level {avg_anxiety:.2f} and depression level {avg_depression:.2f}.")
else:
    print("2. No specific cluster showed dominant presence of this distortion.")
print(f"3. Students in this group may benefit from targeted CBT interventions addressing {most_observed_distortion}.")
print("4. Suggested strategies include thought-challenging exercises, journaling, and mindfulness.")
print("5. Institutions should implement regular mental health screening and workshops to support affected students.")
