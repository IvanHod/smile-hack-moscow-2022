import json
import networkx as nx
import numpy as np
import pandas as pd


def remove_q(seq_list):
    return json.loads(seq_list.replace("'", '"').strip())


def get_couples(seq_list):
    couples = []
    max_indx = len(seq_list) - 1
    for i in range(max_indx):
        couples.append([seq_list[i], seq_list[i + 1]])
    return couples


def get_edges(edge):
    return (edge['Origin'], edge['Source'], {'weight': edge['Weight']})


def fit_graph(sequences_matrix_path: str, sequences_traintest_path: str) -> pd.DataFrame:
    edges_dict = {}

    def fill_edges_dict(edges):
        for edge in edges:
            edge = tuple(edge)

            if edge not in edges_dict:
                edges_dict[edge] = 1
            else:
                edges_dict[edge] += 1

    df = pd.read_csv(sequences_matrix_path, sep='\t')
    df['SESSIONS_SEQUENCES'] = df['SESSIONS_SEQUENCES'].apply(remove_q)
    df['SESSIONS_COUPLES'] = df['SESSIONS_SEQUENCES'].apply(get_couples)
    __ = df['SESSIONS_COUPLES'].apply(fill_edges_dict)

    tmp = list(map(lambda kv: (kv[0][0], kv[0][1], kv[1]), edges_dict.items()))
    edges_df = pd.DataFrame(tmp, columns=['Origin', 'Source', 'Weight'])
    del tmp

    edges_df = edges_df.loc[edges_df['Weight'] > 1]
    edges_df['edges_dict'] = edges_df.apply(lambda row: get_edges(row), axis=1)

    G = nx.Graph(edges_df['edges_dict'].values.tolist())
    remove = [node for node, degree in dict(G.degree()).items() if degree < 2]
    G.remove_nodes_from(remove)
    res_dict = dict(map(lambda n: (n[0], {'degree': n[1]}), dict(G.degree()).items()))
    clusters = nx.clustering(G)

    for nodes, w in nx.get_edge_attributes(G, 'weight').items():
        for n in nodes:
            if n in res_dict:
                if 'weight' in res_dict[n]:
                    res_dict[n]['weight'] += w
                else:
                    res_dict[n]['weight'] = w
            else:
                res_dict[n]['weight'] = w

            res_dict[n]['cluster'] = clusters.get(n)

    df_with_client = pd.read_csv(sequences_traintest_path, sep='\t')
    client_df = df_with_client.to_dict('records')

    result = {}
    for row in client_df:
        length = len(row['SEQUENCE'])
        if length:
            cluster = None
            weight = degree = 0
            SEQUENCE = json.loads(row['SEQUENCE'].replace("'", '"'))
            for node in SEQUENCE:
                if node in res_dict:
                    weight += res_dict[node].get('weight', 0)
                    degree += res_dict[node].get('degree', 0)
                    cluster = res_dict[node].get('cluster')
            result[row['CLIENT_ID']] = {'weight': int(weight / length), 'degree': int(degree / length, ),
                                        'cluster': cluster}

    res_records = []
    for CLIENT_ID, data in result.items():
        row = {'CLIENT_ID': CLIENT_ID, 'weight': data['weight'], 'degree': data['degree'], 'cluster': data['cluster']}
        res_records.append(row)

    return pd.DataFrame.from_records(res_records)
