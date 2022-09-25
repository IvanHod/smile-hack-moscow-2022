import argparse
import zipfile
import os
import pandas as pd
from pathlib import Path


from src import VkSmileModel

parser = argparse.ArgumentParser(description='Smile baseline')
parser.add_argument('--output_folder', default='output_data', type=str)
parser.add_argument('--input_folder', default='input_folder', type=str)

args = parser.parse_args()


def result(input_folder, output_folder):
    files = list(filter(lambda f: not f.startswith('.'), os.listdir(input_folder)))
    if len(files) == 0:
        raise Exception('No input data')
    input_file = files[0]

    if '.zip' in input_file:
        with zipfile.ZipFile(f'{input_folder}/{input_file}', 'r') as zip_ref:
            zip_ref.extractall(input_folder)
            os.remove(f'{input_folder}/{input_file}')
            input_file = list(filter(lambda f: not f.startswith('.'), os.listdir(input_folder)))[0]

    df_test = pd.read_csv(f'{input_folder}/{input_file}', sep='\t')
    proba = apply_model(df_test)

    proba_df = pd.DataFrame(proba, columns=['proba_1', 'proba_2'])
    proba_df['CLIENT_ID'] = df_test['CLIENT_ID'].values
    proba_df = proba_df[['CLIENT_ID', 'proba_1', 'proba_2']]

    proba_df.to_csv(f'/code/{output_folder}/test_eval.csv', sep='\t', index=False)


def apply_model(df: pd.DataFrame):
    y = None  # by default y may be not to be

    path = Path(__file__).parent / 'src' / 'models'

    if not (path / 'w2w_seq_all.pkl').exists():
        os.system("wget 'https://smile.actcognitive.org/media/w2w_seq_all.pkl'")

    smile = VkSmileModel(str(path))

    proba = smile.predict_proba(df, friends_path='', hash_path='')

    return proba


if __name__ == '__main__':
    output_folder = args.output_folder
    input_folder = args.input_folder
    result(input_folder, output_folder)
