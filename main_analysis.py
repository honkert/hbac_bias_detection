import pickle as pkl
import pandas as pd
import lightgbm as lgb
from scipy import stats
from scipy.stats import ttest_ind
from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import hbac_kmeans
from sklearn import metrics
# import shap
from statistics import mean

class HBAC_analyser:
    def __init__(self):
        self.mean_clusters = pd.DataFrame()
        self.all_unscaled_discriminated = None
        self.all_unscaled_discriminated = None

    # create parallel coordinate plot
    def create_parallel_coord_plot(self, df, title):
        # creating df for parallel coordinate plot
        df_parallel = df.copy()
        df_parallel['relative_difference'] = df_parallel['difference'] / df_parallel['unscaled_remaining_mean']
        df_parallel.drop(
            ['unscaled_discriminated_mean', 'unscaled_remaining_mean', 'difference', 'abs_relative_difference',
             'welch_statistic', 'p_value'], axis=1, inplace=True)
        # df_parallel.drop(['unscaled_discriminated_mean','unscaled_remaining_mean','difference','welch_statistic','p_value'],axis=1,inplace=True)
        df_parallel_transpose = df_parallel.T
        df_parallel_transpose['index'] = df_parallel_transpose.index

        # create parallel coordinate plot
        disc_plot = parallel_coordinates(df_parallel_transpose, 'index', color=('#ff003c'), axvlines=False)
        plt.xticks(rotation=90, fontsize=10)
        plt.legend(loc="upper left")
        plt.xlabel("Features")
        plt.ylabel("Relative difference")
        plt.grid(linewidth=0.1)
        plt.title(title)
        plt.show()

    def plot_distributions(self):
        ##### Plot distribution discriminated cluster vs remaining data
        unscaled_remaining = self.all_unscaled_remaining.copy()
        unscaled_discriminated = self.all_unscaled_discriminated.copy()
        unscaled_discriminated['discriminated'] = 1
        unscaled_remaining['discriminated'] = 0
        unscaled_combined = pd.concat([unscaled_discriminated, unscaled_remaining])
        features_sorted = list(self.mean_clusters.T.columns)
        features_sorted.extend(['discriminated'])
        unscaled_combined = unscaled_combined[features_sorted]
        pd.to_pickle(unscaled_combined, "all_unscaled_combined.pkl")
        unscaled_combined = pd.concat([unscaled_discriminated, unscaled_remaining])
        for col in self.mean_clusters.T.columns:
            if len(unscaled_remaining[col].unique()) > 15:
                fig, ax = plt.subplots()
                for a in [unscaled_remaining[col], unscaled_discriminated[col]]:
                    sns.distplot(a, ax=ax)
                    plt.legend(labels=["Remaining data", "Discriminated cluster"])
                plt.show()

            # Quick fix to enter categorical
            if len(unscaled_remaining[col].unique()) <= 15 and col != 'discriminated':
                fig, ax = plt.subplots()
                relative_count = (unscaled_combined.groupby(['discriminated', col]).size() /
                                  unscaled_combined.groupby(['discriminated']).size()).reset_index().rename(
                    {0: 'Relative frequency'}, axis=1)
                sns.barplot(x=col, hue='discriminated', y='Relative frequency', data=relative_count, ax=ax)
                sns.despine(fig)
                plt.show()


    def run_hbac(self,df, num_runs = 1, error_scaling_factor=0.8, plot_clusters=True,plot_distributions=True, parallel_plot=True):
        avg_bias = []
        avg_sillh = []
        avg_number_of_clusters = []
        for run_i in range(1, num_runs + 1):
            results = hbac_kmeans.hbac_kmeans(df, error_scaling_factor, show_plot=plot_clusters)
            c, max_neg_bias = hbac_kmeans.get_max_bias_cluster(results)
            avg_bias.append(max_neg_bias)
            avg_sillh.append(metrics.silhouette_score(
                results.drop(['clusters', 'new_clusters', 'predicted_value', 'true_value', 'errors'], axis=1),
                results['clusters']))
            avg_number_of_clusters.append(len(results['clusters'].unique()))
            discriminated_cluster = results[results['clusters'] == c]
            unscaled_discriminated = df.loc[discriminated_cluster.index, :]
            unscaled_remaining = df.drop(discriminated_cluster.index)
            if run_i == 1:
                self.all_unscaled_discriminated = unscaled_discriminated.copy()
                self.all_unscaled_remaining = unscaled_remaining.copy()
            else:
                self.all_unscaled_discriminated = pd.concat([self.all_unscaled_discriminated, unscaled_discriminated])
                self.all_unscaled_remaining = pd.concat([self.all_unscaled_remaining, unscaled_remaining])
        # Welch's test for differences discrimninated vs remaining data
        significant_features = []
        welch_T = []
        p_values = []
        for i in self.all_unscaled_remaining:
            welch_i = stats.ttest_ind(self.all_unscaled_discriminated[i], self.all_unscaled_remaining[i], equal_var=False)
            welch_T.append(welch_i.statistic)
            p_values.append(welch_i.pvalue)
            if welch_i.pvalue <= 0.05:
                significant_features.append(i)
        self.mean_clusters['unscaled_discriminated_mean'] = self.all_unscaled_discriminated.mean()
        self.mean_clusters['unscaled_remaining_mean'] = self.all_unscaled_remaining.mean()
        self.mean_clusters['difference'] = self.mean_clusters['unscaled_discriminated_mean'] - self.mean_clusters['unscaled_remaining_mean']
        self.mean_clusters['abs_relative_difference'] = abs(self.mean_clusters['difference'])
        self.mean_clusters['abs_relative_difference'] = self.mean_clusters['abs_relative_difference'] / self.mean_clusters[
            'unscaled_remaining_mean']
        self.mean_clusters['welch_statistic'] = welch_T
        self.mean_clusters['p_value'] = p_values
        # SORT ON: absolute relative mean difference between means
        self.mean_clusters = self.mean_clusters.sort_values(by='abs_relative_difference', ascending=False)


        if parallel_plot:
            self.create_parallel_coord_plot(self.mean_clusters,
                                   "Relative difference between discriminated and remaining data")
        if plot_distributions:
            self.plot_distributions()

        print('Averages results of {} runs of HBAC'.format(num_runs))
        print('Average maximun negative biased cluster: ', mean(avg_bias), '±', np.std(avg_bias), ', min=', min(avg_bias))
        print('Average number of clusters: ', mean(avg_number_of_clusters), '±', np.std(avg_number_of_clusters))
        print('Average Sillhouette score: ', mean(avg_sillh), '±', np.std(avg_sillh), ', max=', max(avg_sillh))

        return self.mean_clusters

