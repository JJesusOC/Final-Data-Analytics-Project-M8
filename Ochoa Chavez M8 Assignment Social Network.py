import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import warnings
from itertools import combinations
from networkx.algorithms.community import greedy_modularity_communities

# Loads and pepares the dataset
df = pd.read_csv("players_22.csv")
df.head()
df.shape
df.info()
 
df.isna().mean() * 100

# This section separates the numeric and categoricall data and add descriptive statistics.
num_cols = df.select_dtypes(include = [np.number])
cat_cols = df.select_dtypes(exclude = [np.number])
print(num_cols.describe())
print(num_cols.median())

# This portion find the outliers in the dataset. 
Q1 = num_cols.quantile(0.25)
Q3 = num_cols.quantile(0.75)
IQR = Q3 - Q1
outliers = (num_cols < (Q1 - 1.5 * IQR)) | (num_cols > (Q3 + 1.5 * IQR))
print(outliers.sum())


# To keep the graph readable, we will focus on trhe top 500 players by overall rating.
cols = ['short_name', 'club_name', 'nationality_name', 'overall', 'value_eur', 'league_name']

df = df[cols].dropna(subset = ['club_name', 'short_name'])
df = df.sort_values('overall', ascending = False).head(500).reset_index(drop = True)

print(f"Players loaded  : {len(df)}")
print(f"Unique clubs    : {df['club_name'].nunique()}")
print(f"Unique leagues  : {df['league_name'].nunique()}")
print(f"Unique nations  : {df['nationality_name'].nunique()}")


'''
Building the Social Network:
We will create a player network based nodes that represent players and edges that two players who share the same club.
Players who play for the same club will be connected, similar to how friendship graphs work.
'''
G = nx.Graph()

for _, row in df.iterrows():
        G.add_node(row['short_name'],
               club        = row['club_name'],
               nationality = row['nationality_name'],
               overall     = row['overall'],
               league      = row['league_name'])

for club, group in df.groupby('club_name'):
    players_in_club = group['short_name'].tolist()
    if len(players_in_club) > 1:
        for p1, p2 in combinations(players_in_club, 2):
            G.add_edge(p1, p2, club = club)

print(f"\nNodes (players)  : {G.number_of_nodes()}")
print(f"Edges (co-club)  : {G.number_of_edges()}")
print(f"Network density  : {nx.density(G):.4f}")
print(f"Is connected     : {nx.is_connected(G)}")

# This section deals with descriptive statistics of the network 
# (ie. how many other players each player is connected to).
degrees = [d for _, d in G.degree()]
print(f"\nDegree — Mean : {np.mean(degrees):.2f}")
print(f"Degree — Max  : {np.max(degrees)}")
print(f"Degree — Min  : {np.min(degrees)}")
print(f"Degree — Std  : {np.std(degrees):.2f}")



'''
vizulization 1: Bar Chart
In this vizualization, we compare how connected players are across the network.
Most players are connected to a similar amount compared to others,
but players in larger clubs (like Real Madrid, Barcelona, etc.) will have more connections.
'''
plt.figure(figsize = (8, 6))
sns.histplot(degrees, bins = 30, color = "#4C72B0", edgecolor = 'white')
plt.title("Degree Distribution — FIFA Player Network")
plt.xlabel("Degree (number of co-club connections)")
plt.ylabel("Number of Players")
plt.tight_layout()
plt.savefig('sna_viz1_degree_distribution.png', dpi = 150)
plt.show()



'''
Centrality Measures:

We will calculate three centrality measures to identify key players in the network:
1. Degree Centrality: How many connections a player has. 
2. Betweenness Centrality: How often a player lies on the shortest path between other players. 
3. PageRank: A measure of influence based on the connections of a player and their neighbors. 

Each meausure gives us a different storty about the player's importance in the network.
'''

degree_cent      = nx.degree_centrality(G)
betweenness_cent = nx.betweenness_centrality(G, k = 200, normalized = True, seed = 0)
pagerank_cent    = nx.pagerank(G, alpha = 0.85)

centrality_df = pd.DataFrame({
    'player'      : list(degree_cent.keys()),
    'degree'      : list(degree_cent.values()),
    'betweenness' : [betweenness_cent[p] for p in degree_cent.keys()],
    'pagerank'    : [pagerank_cent[p]    for p in degree_cent.keys()]
})

centrality_df = centrality_df.merge(
    df[['short_name', 'club_name', 'nationality_name', 'overall', 'league_name']],
    left_on = 'player', right_on = 'short_name', how = 'left'
).drop(columns = 'short_name')

centrality_df = centrality_df.sort_values('pagerank', ascending = False).reset_index(drop = True)

print("\nTop 10 Most Influential Players (PageRank):")
print(centrality_df[['player', 'club_name', 'league_name',
                      'overall', 'degree', 'betweenness', 'pagerank']].head(10).to_string(index = False))


'''
Vizualization 2: Top 20 players by each centrality measure. 
In this vizualization, we will compare the top 20 players by all three centrality measures. 
Each measure will highlight different players as influential. 
''' 
fig, axes = plt.subplots(1, 3, figsize = (18, 6))

metrics = [
    ('degree',      '#4C72B0', 'Degree Centrality'),
    ('betweenness', '#E53935', 'Betweenness Centrality'),
    ('pagerank',    '#43A047', 'PageRank')
]

for ax, (metric, color, title) in zip(axes, metrics):
    top20 = centrality_df.nlargest(20, metric)[['player', metric]].sort_values(metric)
    ax.barh(top20['player'], top20[metric], color = color, alpha = 0.85)
    ax.set_title(title, fontweight = 'bold')
    ax.set_xlabel('Score')
    ax.tick_params(labelsize = 8)

plt.suptitle("Top 20 Influential Players — Centrality Measures", fontsize = 13, fontweight = 'bold')
plt.tight_layout()
plt.savefig('sna_viz2_centrality_bars.png', dpi = 150)
plt.show()


'''
Community Detection:
We will find natural communities within the player network.
These communities should roughly align with leagues or national groups,
as players in the same clubs naturally cluster together.
'''
communities  = list(greedy_modularity_communities(G))
print(f"\nCommunities found: {len(communities)}")

community_map = {}
for i, comm in enumerate(communities):
    for player in comm:
        community_map[player] = i

nx.set_node_attributes(G, community_map, 'community')

comm_sizes = pd.Series(community_map).value_counts().sort_index()
print("\nTop community sizes:")
print(comm_sizes.head(10).to_string())

for comm_id in comm_sizes.head(5).index:
    members    = [p for p, c in community_map.items() if c == comm_id]
    subset     = centrality_df[centrality_df['player'].isin(members)]
    if len(subset) > 0 and subset['league_name'].notna().any():
        top_league = subset['league_name'].value_counts().index[0]
    else:
        top_league = 'N/A'
    print(f"  Community {comm_id}: {len(members)} players | Dominant league: {top_league}")


'''
Vizualization 3: Network Graph
In this vizualization, we will vizualize the top 100 players as a network.
Node size will be based on PageRank, so bigger nodes means more influence.
The colors respresent the community the player belongs to. 
'''
top_players      = centrality_df.head(100)['player'].tolist()
sub              = G.subgraph(top_players).copy()
node_communities = [community_map.get(n, 0) for n in sub.nodes()]
node_sizes       = [pagerank_cent.get(n, 0.001) * 15000 for n in sub.nodes()]
pos              = nx.spring_layout(sub, seed = 0, k = 0.6)

plt.figure(figsize = (14, 10))
nx.draw_networkx_edges(sub, pos, alpha = 0.2, edge_color = 'gray', width = 0.5)
scatter = nx.draw_networkx_nodes(sub, pos,
                                  node_color = node_communities,
                                  node_size  = node_sizes,
                                  cmap       = plt.cm.tab20,
                                  alpha      = 0.85)
nx.draw_networkx_labels(sub, pos, font_size = 6, font_color = 'black')
plt.colorbar(scatter, label = 'Community ID', shrink = 0.6)
plt.title("FIFA Player Influence Network — Top 100 Players\n(Node size = PageRank, Color = Community)",
          fontsize = 13, fontweight = 'bold')
plt.axis('off')
plt.tight_layout()
plt.savefig('sna_viz3_network_graph.png', dpi = 150)
plt.show()



'''
Vizualization 4: Heatmap
In this vizualization, we will correlate the three centrality measures and compare them to each other and to the overall rating of the players.
Warmer colors indicate stronger agreement between mreasures on who is influential.
'''
heat_cols = ['degree', 'betweenness', 'pagerank', 'overall']
corr      = centrality_df[heat_cols].corr(method = 'spearman')

plt.figure(figsize = (8, 6))
sns.heatmap(corr, annot = True, fmt = '.2f', cmap = 'coolwarm',
            center = 0, vmin = -1, vmax = 1, linewidths = 0.5)
plt.title("Centrality Correlation Heatmap")
plt.tight_layout()
plt.savefig('sna_viz4_centrality_heatmap.png', dpi = 150)
plt.show()


# Final Network Summary
print(f"\nTotal players (nodes)     : {G.number_of_nodes()}")
print(f"Total connections (edges) : {G.number_of_edges()}")
print(f"Network density           : {nx.density(G):.4f}")
print(f"Communities detected      : {len(communities)}")
print(f"\nMost influential player   : {centrality_df.iloc[0]['player']}")
print(f"  Club                    : {centrality_df.iloc[0]['club_name']}")
print(f"  League                  : {centrality_df.iloc[0]['league_name']}")
print(f"  PageRank score          : {centrality_df.iloc[0]['pagerank']:.6f}")


