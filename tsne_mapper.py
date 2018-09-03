"""
@author: Noel
@date: {13/08/2018}
"""

import pandas as pd
import numpy as np
import kmapper as km
import sklearn
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import igraph as ig


def get_cluster_summary(player_list, average_mean, average_std, dataset, columns):
    # Compare players against the average and list the attributes that are above and below the average

    cluster_mean = np.mean(dataset.iloc[player_list].values, axis=0)
    diff = cluster_mean - average_mean
    std_m = np.sqrt((cluster_mean - average_mean) ** 2) / average_std

    stats = sorted(zip(columns, cluster_mean, average_mean, diff, std_m), key=lambda x: x[4], reverse=True)
    above_stats = [a[0] + ': ' + f'{a[1]:.2f}' for a in stats if a[3] > 0]
    below_stats = [a[0] + ': ' + f'{a[1]:.2f}' for a in stats if a[3] < 0]

    # Create a string summary for the tooltips
    cluster_summary = 'Above Mean:<br>' + '<br>'.join(above_stats[:5]) + \
                      '<br><br>Below Mean:<br>' + '<br>'.join(below_stats[-5:])

    return cluster_summary


np.random.seed(1234)

# Load dataset
df = pd.read_csv("DetailedPlayerStats.csv", encoding='ANSI')
df.dropna(inplace=True)

# Map the positions to shorthand
positions = {'Goalkeeper': 'GK', 'Defender': 'DEF', 'Forward': 'FWD', 'Midfielder': 'MID'}
df['Position'] = df['Position'].map(lambda x: positions[x])
df['Identifier'] = df['WebName'] + '-' + df['Team'] + '-' + df['Position']

# Remove redundant features
to_drop = ['Id', 'WebName', 'FirstName', 'SecondName', 'DreamTeam', 'SelectedBy',
           'Position', 'Team', 'FixtureWeek1', 'Opponent1', 'HomeAway1',
           'FixtureWeek2', 'Opponent2', 'HomeAway2', 'FixtureWeek3', 'Opponent3',
           'HomeAway3', 'GameWeek', 'Cost', 'NetTransfers', 'GoalsConceded']
features = [col for col in df.columns if col not in to_drop]
df = df[features]


# Function to convert all stats to per 90
def min_converter(row):
    mins = row['MinutesPlayed']
    if mins == 0:
        return row
    ret = []
    for col in row.index:
        if col not in ['Identifier', 'Points', 'MinutesPlayed']:
            ret.append(row[col] / float(mins) * 90.0)
        else:
            ret.append(row[col])
    return ret


# Convert all values to per 90 mins
df = df.apply(min_converter, axis=1, result_type='broadcast')
df = df.apply(pd.to_numeric, errors='ignore')
# df.to_csv('PlayerStats_Season1819.csv', index=False, encoding='ANSI')

# Filter out players who didn't play more than 25 mins
df = df[df['MinutesPlayed'] > 25]

# Drop the mins played - want to base players on their performance rather than time on the pitch
df.drop('MinutesPlayed', axis=1, inplace=True)
df.drop('CleanSheet', axis=1, inplace=True)

# Add some derived stats
df['PassAccuracy'] = (df['PassesCompleted'] / df['AttemptedPasses']).replace(np.nan, 0)

# Get average stats per game per player
g = df.groupby('Identifier')
df2 = g.aggregate(np.mean)

# Drop fantasy related colunms - we only want to cluster players based on their in-game performance
X = df2[[col for col in df2.columns if col not in ['Points', 'Bonus']]]
names = X.index.values
X.index = [i for i in range(X.shape[0])]

# Get averages for each stat
means = np.mean(X.values, axis=0)
std_dev = np.std(X.values, axis=0)

# Initialise mapper and create lens using TSNE
mapper = km.KeplerMapper(verbose=0)
lens = mapper.fit_transform(X.values, projection=sklearn.manifold.TSNE(), scaler=None)

# Create the graph of the nerve of the corresponding pullback
graph = mapper.map(lens, X.values,
                   # clusterer=sklearn.cluster.DBSCAN(eps=0.3, min_samples=1),
                   clusterer=sklearn.cluster.KMeans(n_clusters=2, random_state=1234),
                   nr_cubes=25, overlap_perc=0.5)

# Get the players per cluster and overall cluster stats
node_dict = {}
node_list = []
node_stats = []
for node in graph['nodes']:
    node_list.append(node)
    players = [names[i] for i in graph['nodes'][node]]
    node_dict[node] = players
    node_stats.append(get_cluster_summary(graph['nodes'][node], means, std_dev, X, X.columns))

# mapper.visualize(graph, path_html="premier_league_players_2.html",
#                  title="Player Data", inverse_X=X.values, inverse_X_names=names, custom_tooltips=names)


# Add the edges to a list for passing into iGraph:
edge_list = []
for node in graph['links']:
    for nbr in graph['links'][node]:
        # Need to base everything on indices for igraph
        edge_list.append((node_list.index(node), node_list.index(nbr)))

n_nodes = len(node_list)
G = ig.Graph(n_nodes)

G.add_edges(edge_list)

avg_points = []
for node in G.vs.indices:
    avg_points.append(np.average([df2.iloc[i]['Points'] for i in graph['nodes'][node_list[node]]]))
    G.vs[node]['size'] = 2 * int(np.log(len(node_dict[node_list[node]]) + 1) + 1)
    # G.vs[node]['size'] = len(node_dict[node_list[node]])

links = G.get_edgelist()
layt = G.layout('fr')

N = len(layt)
Xnodes = [layt[k][0] for k in range(N)]  # x-coordinates of nodes
Ynodes = [layt[k][1] for k in range(N)]  # y-coordnates of nodes

Xedges = []
Yedges = []
for e in links:
    Xedges.extend([layt[e[0]][0], layt[e[1]][0], None])
    Yedges.extend([layt[e[0]][1], layt[e[1]][1], None])

edges_trace = dict(type='scatter',
                   x=Xedges,
                   y=Yedges,
                   mode='lines',
                   line=dict(color='rgb(200,200,200)',
                             width=0.3),
                   hoverinfo='none')

nodes_trace = dict(type='scatter',
                   x=Xnodes,
                   y=Ynodes,
                   mode='markers',
                   opacity=0.8,
                   marker=dict(symbol='dot',
                               colorscale='Viridis',
                               showscale=True,
                               reversescale=False,
                               color=avg_points,
                               size=G.vs['size'],
                               line=dict(color='rgb(200,200,200)',
                                         width=0.5),
                               colorbar=dict(thickness=20,
                                             ticklen=4)),
                   text=[],
                   hoverinfo='text')

axis = dict(showline=False,
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            title='')

layout = dict(title='Test',
              font=dict(size=12),
              showlegend=False,
              autosize=False,
              width=700,
              height=700,
              xaxis=dict(axis),
              yaxis=dict(axis),
              hovermode='closest',
              plot_bgcolor='rgba(20,20,20, 0.8)')

# Add tooltips
nodes = G.vs.indices
div = '<br>-------<br>'

sizes = []
trace = []
for node in nodes:
    node_name = node_list[node]
    players = node_dict[node_name]
    sizes.append(len(players))
    node_info = node_name + div + '<br>'.join(players) + div + node_stats[node]
    nodes_trace['text'] += tuple([node_info])
    trace += tuple([node_info])

plot(dict(data=[edges_trace, nodes_trace], layout=layout))
