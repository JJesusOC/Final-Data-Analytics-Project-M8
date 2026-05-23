import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score, f1_score)
 
# Loads and pepares the dataset
df = pd.read_csv("results.csv")
df.head()
df.shape
df.info()

df.isna().mean() * 100

# This section separartes numeric and categorical data and adds descriptive statistics. 
num_cols = df.select_dtypes(include=[np.number])
cat_cols = df.select_dtypes(exclude=[np.number])
print(num_cols.describe())
print(num_cols.median())

# This portion helps with finding the outliers in the dataset
Q1 = num_cols.quantile(0.25)
Q3 = num_cols.quantile(0.75)
IQR = Q3 - Q1
outliers = (num_cols < (Q1 - 1.5 * IQR)) | (num_cols > (Q3 + 1.5 * IQR))
print(outliers.sum())

# From the home team's perspective, we create a target variable, decided by a win, draw or a loss. 
def get_result(row):
    if row['home_score'] > row['away_score']:
        return 'Win'
    elif row['home_score'] < row['away_score']:
        return 'Loss'
    else:
        return 'Draw'
 
df['result'] = df.apply(get_result, axis = 1)
 
print(df['result'].value_counts())

# Here we create features that may provode to be usefule for the model.
# Rows with missing values are dropped. 
df['goal_diff']   = df['home_score'] - df['away_score']
df['total_goals'] = df['home_score'] + df['away_score']
df['neutral']     = df['neutral'].astype(int)

df = df[['home_team', 'home_score', 'away_score', 'goal_diff',
          'total_goals', 'neutral', 'result']].dropna()

print(f"Clean dataset shape: {df.shape}")

le = LabelEncoder()
df['result_encoded'] = le.fit_transform(df['result'])


'''
Vizualization 1: Bar Chart
In this vizualization, we compare how many matches end in either a win, loss or draw for the home team.
Home advantage is a common phenomenon in soccer. Wins are expected most common due to this phenonomenon.
'''

result_counts = df['result'].value_counts()

plt.figure(figsize = (8, 6))
sns.barplot(x = result_counts.index, y = result_counts.values, color = "#4C72B0")
plt.title("Match Outcome Distribution (Home Team Perspective)")
plt.xlabel("Result")
plt.ylabel("Number of Matches")
plt.tight_layout()
plt.savefig('viz1_outcome_distribution.png', dpi = 150)
plt.show()


'''
Vizualization 2: Scatter Plot
In this vizualization, we plot home score against away score colored by match result.
The highest scoring matches are highlghted to reflect the outliers in the dataset.
'''
df_plot = df[['home_score', 'away_score', 'result', 'total_goals']].copy()

plt.figure(figsize = (8, 6))
ax = sns.scatterplot(data = df_plot, x = 'home_score', y = 'away_score',
                     hue = 'result', alpha = 0.4, s = 20,
                     palette = {'Win': '#4C72B0', 'Draw': '#FF9800', 'Loss': '#E53935'})

top_outliers = df_plot.nlargest(5, 'total_goals')
for _, r in top_outliers.iterrows():
    ax.annotate(f"{int(r['home_score'])}-{int(r['away_score'])}",
                (r['home_score'], r['away_score']),
                textcoords="offset points", xytext = (5, 5),
                fontsize = 8, alpha = 0.9)

plt.title("Home Score vs Away Score by Match Result")
plt.xlabel("Home Score")
plt.ylabel("Away Score")
plt.tight_layout()
plt.savefig('viz2_score_scatter.png', dpi = 150)
plt.show()


'''
Vizualization 3: Heatmap
In this vizualization, we take a look at how our numeric features correlate with each other.
Warmer colors indicate a stronger positrive correlation, while colder colors indicate weaker/negative correlation.
'''

heat_cols = ['home_score', 'away_score', 'goal_diff', 'total_goals', 'neutral']
corr = df[heat_cols].corr(method='spearman')

plt.figure(figsize = (8, 6))
sns.heatmap(corr, annot = True, fmt = '.2f', cmap = 'coolwarm',
            center = 0, vmin = -1, vmax = 1, linewidths = 0.5)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig('viz3_heatmap.png', dpi = 150)
plt.show()


'''
Technique 1: K-Means Clustering
For this, we group teams by playing style by using the K-Means Clustering technique.
We create team profiles by basing it on their average goals scored, conceded, and win rate.
NOTE: Only teams with at least 10 matches are included to ensure meaningful profiles. 
'''

team_profile = df.groupby('home_team').agg(
    avg_scored    = ('home_score', 'mean'),
    avg_conceded  = ('away_score', 'mean'),
    total_wins    = ('result', lambda x: (x == 'Win').sum()),
    total_matches = ('result', 'count')
).reset_index()

team_profile['win_rate'] = team_profile['total_wins'] / team_profile['total_matches']
team_profile = team_profile[team_profile['total_matches'] >= 10].reset_index(drop=True)

x_cluster = StandardScaler().fit_transform(
    team_profile[['avg_scored', 'avg_conceded', 'win_rate']])

# The elbow methos is used to determine the best number of clusters. 
inertias = []
for k in range(2, 10):
    km = KMeans(n_clusters = k, random_state = 0, n_init = 10)
    km.fit(x_cluster)
    inertias.append(km.inertia_)

plt.figure(figsize = (8, 6))
plt.plot(range(2, 10), inertias, marker = 'o', color = '#4C72B0', linewidth = 2)
plt.axvline(x = 4, color = 'red', linestyle = '--', label='Chosen k = 4')
plt.title("Elbow Method — Choosing Optimal K for Clustering")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.legend()
plt.tight_layout()
plt.savefig('viz4_elbow.png', dpi = 150)
plt.show()

km_final = KMeans(n_clusters = 4, random_state = 0, n_init = 10)
team_profile['cluster'] = km_final.fit_predict(x_cluster)

print("\nCluster Profiles:")
print(team_profile.groupby('cluster')[
    ['avg_scored', 'avg_conceded', 'win_rate']
].mean().round(3))

# Scatter plot is used to visualize the clusters.
plt.figure(figsize = (8, 6))
colors = ['#E53935', '#1E88E5', '#43A047', '#FB8C00']
for c in range(4):
    subset = team_profile[team_profile['cluster'] == c]
    plt.scatter(subset['avg_scored'], subset['avg_conceded'],
                label = f'Cluster {c}', color = colors[c], alpha = 0.75, s = 60)

plt.title("Team Clusters: Avg Goals Scored vs Avg Goals Conceded")
plt.xlabel("Avg Goals Scored")
plt.ylabel("Avg Goals Conceded")
plt.legend(title = "Cluster")
plt.tight_layout()
plt.savefig('viz5_clusters.png', dpi = 150)
plt.show()


'''
Technique 2: Logistic Regression
The purpose of this technique is to prediect match outcomes using goals and match features.
It uses a pipeline to handle missing values, scale features, and train a logistic regression model.
'''
feature_cols = ['total_goals', 'neutral']

x = df[feature_cols]
y_clf = df['result_encoded']

x_train, x_test, y_train, y_test = train_test_split(
    x, y_clf, test_size = 0.4, random_state = 0)

log_model = Pipeline([
    ('imputer', SimpleImputer(strategy = 'median')),
    ('scaler',  StandardScaler()),
    ('model',   LogisticRegression(
        max_iter = 2500,
        solver = 'lbfgs',
        class_weight = 'balanced',
        random_state = 0
    ))
])

log_model.fit(x_train, y_train)
y_pred_lr = log_model.predict(x_test)

print("\nLogistic Regression Results:")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
print(f"Accuracy:  {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_lr, average='weighted'):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_lr, average='weighted'):.4f}")
print(f"F1-score:  {f1_score(y_test, y_pred_lr, average='weighted'):.4f}")


'''
Technique 3: Random Forest 
As we used logistic regression to predict match outcomes, 
we also use a random forest classifier to see if we can improve the performance.
We will be comparing both models. 
'''
rf_model = Pipeline([
    ('imputer', SimpleImputer(strategy = 'median')),
    ('scaler',  StandardScaler()),
    ('model',   RandomForestClassifier(
        n_estimators = 100,
        class_weight = 'balanced',
        random_state = 0
    ))
])

rf_model.fit(x_train, y_train)
y_pred_rf = rf_model.predict(x_test)

print("\nRandom Forest Results:")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print(f"Accuracy:  {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_rf, average = 'weighted'):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_rf, average = 'weighted'):.4f}")
print(f"F1-score:  {f1_score(y_test, y_pred_rf, average = 'weighted'):.4f}")

# This is the feature importance plot for the random forest model. 
# It wii show which features were most important for the model.
importances = pd.Series(
    rf_model.named_steps['model'].feature_importances_,
    index=feature_cols
).sort_values(ascending=True)

plt.figure(figsize = (8, 5))
importances.plot(kind = 'barh', color = '#4C72B0')
plt.title("Random Forest — Feature Importances")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig('viz6_rf_importance.png', dpi = 150)
plt.show()


# Model compairson summary
models     = ['Logistic Regression', 'Random Forest']
accuracies = [accuracy_score(y_test, y_pred_lr), accuracy_score(y_test, y_pred_rf)]

plt.figure(figsize = (7, 4)) 
sns.barplot(x = models, y = accuracies, palette = ['#4C72B0', '#43A047'])
plt.ylim(0, 1)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.01, f'{v:.2%}', ha = 'center', fontweight = 'bold')
plt.tight_layout()
plt.savefig('viz7_model_comparison.png', dpi = 150)
plt.show()


