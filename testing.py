from pathlib import Path

import pandas as pd

from src import VkSmileModel

path = Path(__file__).parent / 'src'
VkSmileModel.load_default_models(current_dir=path)

path = Path(__file__).parent / 'src' / 'models'
smile = VkSmileModel(str(path))


df = pd.read_csv('input_data/df_test.csv', sep='\t')
proba = smile.predict_proba(df, friends_path='', hash_path='input_data/hashes.csv')
