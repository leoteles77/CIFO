import pandas as pd
import matplotlib.pyplot as plt

def transform_to_df(file):
    df = pd.read_csv(file)
    df = df[['gen', 'bestfitness']]
    df = df.groupby(['gen']).mean()
    df['gen'] = df.index
    return df

# List of CSV files
files = ['results/ex1_inv_blen.csv', 'results/exp2_inv_arit.csv','results/exp3_inv_geo.csv','results/exp4_scrab_blen.csv','results/exp5_scrab_arit.csv','results/exp6_scrab_geo.csv','results/exp7_geo_blen.csv','results/exp8_geo_arit.csv','results/exp9_geo_geo.csv']
legends = ['EXP1','EXP2','EXP3','EXP4','EXP5','EXP6','EXP7','EXP8','EXP9']
files_elit = ['results/exp2_inv_arit.csv','results/exp6_scrab_geo.csv','results/exp10_inv_arit_elist_FALSE.csv','results/exp11_inv_arit_elist_5.csv','results/exp12_scrab_geo_elist_FALSE.csv','results/exp13_scrab_geo_elist_5.csv']
legends_elit = ['EXP2_1','EXP6_1','EXP2_FALSE','EXP2_5','EXP6_FALSE','EXP6_5']


# Plotting configuration
plt.figure(figsize=(10, 6))
plt.xlabel('Generation')
plt.ylabel('Average Best Fitness')
plt.title('GA Performance')

# Plotting data for each file
for file, legend in zip(files_elit, legends_elit):
    data_frame = transform_to_df(file)
    plt.plot(data_frame['gen'], data_frame['bestfitness'], label=legend)

# Display legend and show the plot
plt.legend()
plt.savefig('results/elistism.png')
plt.show()