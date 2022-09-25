import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from . import constants


class VkSmileModel:
    class Models:
        def __init__(self, models_path: str, fit_mode=False):
            with open(f'{models_path}/w2w_seq_all.pkl', 'rb') as f:
                m_w2v = pickle.load(f)

            with open(f'{models_path}/model.pkl', 'rb') as f:
                model = pickle.load(f)

            self.m_w2v = m_w2v
            self.model = model

            self.df_graph = pd.read_csv(f'{models_path}/clients_weight_degree.csv', sep='\t').drop('Unnamed: 0', axis=1,
                                                                                                   errors='ignore')
            self.df_graph = self.df_graph.set_index('CLIENT_ID')

            self.df_friends = None
            if os.path.exists(f'{models_path}/friends_features_result_all.csv'):
                df_friends = pd.read_csv(f'{models_path}/friends_features_result_all.csv', sep='\t')
                df_friends = df_friends.set_index('CLIENT_ID')
                # df_friends = df_friends[df_friends.index.isin(set(df.index))]

                df_friends.columns = list(map(lambda c: f'friend_{c}', df_friends.columns))

                self.df_friends = df_friends

    def __init__(self, models_path: str = 'src/models', fit_mode=False):
        self.models = self.Models(models_path, fit_mode=fit_mode)

    def fit(self, data: pd.DataFrame, friends_path: str, hash_path: str,
            sequences_matrix_path: str):
        pass

    def _fit_graph(self, sequences_matrix_path: str) -> pd.DataFrame:
        pass

    def _fit_w2v(self, sequences_matrix_path: str):
        pass  # Должна возвращать модель

    def _fit_tfidf(self, sequences_matrix_path: str):
        pass  # Должна возвращать модель

    def predict(self, data: pd.DataFrame, friends_path: str, hash_path: str):
        proba = self.predict_proba(data, friends_path, hash_path)
        return (proba[:, 1] > 0.5).astype(int)

    def predict_proba(self, data: pd.DataFrame, friends_path: str, hash_path: str):
        data = data.copy()
        df = data.set_index('CLIENT_ID')

        df = self._process_date(df)

        main_features = ['quarter', 'day', 'dayofweek', 'is_month_start', 'is_month_end'] + constants.main_features_selected
        df = df[main_features]
        print('Features were selected', df.shape)

        df = self._process_friends(df, friends_path=friends_path)
        print('Friends were processed', df.shape)

        df = self._process_w2v(df, hash_path=hash_path)
        print('w2v were processed', df.shape)

        df = self._process_graph(df)
        print('Graph was processed', df.shape)

        df = df.fillna(0).reset_index()

        return self.models.model.predict_proba(df)

    def _merge(self, df1, df2):
        return df1.merge(df2, how='left', left_index=True, right_index=True)

    def _process_date(self, df: pd.DataFrame) -> pd.DataFrame:
        df['RETRO_DT'] = pd.to_datetime(df['RETRO_DT'], format='%Y%m%d')

        df['quarter'] = df['RETRO_DT'].dt.quarter
        df['day'] = df['RETRO_DT'].dt.day
        df['dayofweek'] = df['RETRO_DT'].dt.dayofweek
        df['is_month_start'] = df['RETRO_DT'].dt.is_month_start
        df['is_month_end'] = df['RETRO_DT'].dt.is_month_end

        df = df.drop('RETRO_DT', axis=1)

        return df

    def _process_friends(self, df: pd.DataFrame, friends_path: str):
        if self.models.df_friends is not None:
            df_friends = self.models.df_friends[constants.friends_features_selected]
            return self._merge(df, df_friends)

        df_features_agg = None
        df_last = None

        def to_agg(df_):
            df_agg = df_.drop('FRIEND_ID', axis=1).groupby('CLIENT_ID', as_index=False).aggregate(
                ['mean', 'max', 'count'])
            df_agg.columns = list(map(lambda c: f'{c[0]}_{c[1]}', df_agg.columns))

            df_count = df_.replace(0, np.nan).groupby('CLIENT_ID').count()
            df_count = (df_count / df_count['FRIEND_ID'].values[None, :].T).drop('FRIEND_ID', axis=1)

            df_agg = pd.concat((df_agg, df_count), axis=1)

            df_agg = df_agg[constants.friends_features_selected]
            return df_agg

        ids_set = set(df.index)
        for i, df_100k in enumerate(pd.read_csv(friends_path, sep='\t', chunksize=100000)):
            print(f'Friends processing: {i} iteration')

            if df_last is not None:
                df_100k = pd.concat((df_last, df_100k))

            df_100k = df_100k[df_100k['CLIENT_ID'].isin(ids_set)]

            last_indices = df_100k['CLIENT_ID'] == df_100k['CLIENT_ID'].values[-1]

            df_last = df_100k[last_indices]  # Забываем до следующей итериации
            df_100k = df_100k[~last_indices]

            df_agg = to_agg(df_=df_100k)

            if df_features_agg is None:
                df_features_agg = df_agg
            else:
                df_features_agg = pd.concat((df_features_agg, df_agg))

        if df_last is not None and df_last.shape[0] > 1:
            df_agg = to_agg(df_=df_last)
            df_features_agg = pd.concat((df_features_agg, df_agg))

        df_friends = df_features_agg
        del df_agg, df_last

        return self._merge(df, df_friends)

    def _process_w2v(self, df: pd.DataFrame, hash_path: str) -> pd.DataFrame:
        m_w2v = self.models.m_w2v

        df_seq = pd.read_csv(hash_path, sep='\t').set_index('CLIENT_ID')
        df_seq = df_seq[df_seq.index.isin(set(df.index))]

        zeros_100 = np.zeros(100)

        def hash_to_vec_100(m, val):
            val = json.loads(val.replace('\'', '"'))
            val = list(filter(lambda v: v in m.wv, val))

            if len(val) == 0:
                return zeros_100

            return m.wv[val].mean(axis=0)

        vectors = df_seq['SEQUENCE'].apply(lambda v: hash_to_vec_100(m_w2v, v))
        vectors = np.array(vectors.values.tolist())

        cols = list(map(lambda i: f'w2w-{i}', range(vectors.shape[1])))
        vectors = pd.DataFrame(vectors, columns=cols, index=df_seq.index)

        return self._merge(df, vectors)

    def _process_graph(self, df: pd.DataFrame) -> pd.DataFrame:
        df_graph = self.models.df_graph

        df_graph = df_graph[df_graph.index.isin(set(df.index))]

        return self._merge(df, df_graph)

    @staticmethod
    def load_default_models(current_dir: Path):
        path = current_dir / 'models'

        if not (path / 'w2w_seq_all.pkl').exists():
            os.system("wget -O w2w_seq_all.pkl 'https://smile.actcognitive.org/media/w2w_seq_all.pkl'")
            os.system("mv w2w_seq_all.pkl models/w2w_seq_all.pkl")

        friends_file = current_dir / 'friends_features_result_all.csv'
        if not friends_file.exists():
            friends_archive = current_dir / 'friends_features_result_all.csv.zip'

            if friends_archive.exists():
                os.system(f"unzip {str(friends_archive)}")
