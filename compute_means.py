import pandas as pd
import numpy as np

if __name__ == '__main__':
    df = pd.read_csv('experiments.csv', sep = ';')
    exp_names = pd.unique(df['Name'])
    for name in exp_names:
        for var in ['AP','acc']:
            mean_overall_acc = np.mean([float(v) for v in df[(df['Name'] == name) & (df['run'] != 'mean')][var]])
            df.loc[(df['Name'] == name) & (df['run'] == 'mean'),var] = mean_overall_acc
    df.to_csv('experiments.csv', index = False, sep = ';')
