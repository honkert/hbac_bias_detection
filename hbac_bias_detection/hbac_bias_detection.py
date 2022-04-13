import pickle as pkl
from scipy import stats
from hbac_bias_detection import hbac_kmeans
from hbac_bias_detection.hbac_utils import *
from sklearn import metrics
import shap
from statistics import mean
from sklearn.model_selection import train_test_split
import xgboost as xgb
# from config import MODEL_MODE, print_mode

class HBAC_analyser:
    def __init__(self, selected_features='all'):
        self.mean_clusters = pd.DataFrame()
        self.all_unscaled_discriminated = None
        self.all_unscaled_discriminated = None
        self.all_unscaled_combined = None
        # if isinstance(selected_features, list):
        self.selected_features = selected_features
        self.clustered_data = None
        # from config import MODEL_MODE, print_mode
        # MODEL_MODE = model_mode
        # print_mode = print_results


    # create parallel coordinate plot
    def lineplot_differences(self, title="Relative difference between discriminated and remaining data"
                             , plot_top_features=False):
        # creating df for parallel coordinate plot
        df_parallel = self.mean_clusters.copy()
        if plot_top_features:
            df_parallel = df_parallel.iloc[:plot_top_features, :]

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

    def plot_distributions(self,plot_top_features=False):
        ##### Plot distribution discriminated cluster vs remaining data
        if plot_top_features:
            plot_features = self.mean_clusters.T.columns[:plot_top_features]
        else:
            plot_features = self.mean_clusters.T.columns
        for col in plot_features:
            if len(self.all_unscaled_remaining[col].unique()) > 15:
                fig, ax = plt.subplots()
                for a in [self.all_unscaled_remaining[col], self.all_unscaled_discriminated[col]]:
                    sns.distplot(a, ax=ax)
                    plt.legend(labels=["Remaining data", "Discriminated cluster"])
                plt.show()

            # Quick fix to enter categorical
            if len(self.all_unscaled_remaining[col].unique()) <= 15 and col != 'discriminated':
                fig, ax = plt.subplots()
                relative_count = (self.all_unscaled_combined.groupby(['discriminated', col]).size() /
                                  self.all_unscaled_combined.groupby(['discriminated']).size()).reset_index().rename(
                    {0: 'Relative frequency'}, axis=1)
                sns.barplot(x=col, hue='discriminated', y='Relative frequency', data=relative_count, ax=ax)
                sns.despine(fig)
                plt.show()


    def hbac_on_model(self, model_path,X_test_data,y_test_data,error_scaling_factor=0.8,num_runs = 1,
                      plot_clustering_steps=False, impute_missing_values = False):
        loaded_model = pkl.load(open(model_path, 'rb'))

        # Make predictions and construct df
        predictions = loaded_model.predict(X_test_data)
        df = X_test_data.copy()
        if isinstance(self.selected_features, list):
            df = df[self.selected_features]
        df['predicted_value'] = predictions.copy() #np.asarray(predictions)
        df['true_value'] = y_test_data.copy()
        df['errors'] = df['predicted_value'] - df['true_value']
        df['errors'] = df['errors'] / df['true_value']

        # Impute missing values with MICE-forest
        if impute_missing_values:
            df = self.MICE(df)

        # Run HBAC
        self.run_hbac(df, error_scaling_factor, num_runs, plot_clustering_steps)


    def run_hbac(self,df, error_scaling_factor, num_runs, plot_clustering_steps):

        if isinstance(self.selected_features, list):
            df = df[self.selected_features]

        avg_bias = []
        avg_sillh = []
        avg_number_of_clusters = []
        for run_i in range(1, num_runs + 1):
            results = hbac_kmeans.hbac_kmeans(df, error_scaling_factor, show_plot=plot_clustering_steps)
            self.clustered_data = results
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
        self.all_unscaled_discriminated['discriminated'] = 1
        self.all_unscaled_remaining['discriminated'] = 0
        # Combine discrimnated and remaining df
        self.all_unscaled_combined = pd.concat([self.all_unscaled_discriminated, self.all_unscaled_remaining])
        features_sorted = list(self.mean_clusters.T.columns)
        features_sorted.extend(['discriminated'])
        self.all_unscaled_combined = self.all_unscaled_combined[features_sorted]
        pd.to_pickle(self.all_unscaled_combined, "../all_unscaled_combined.pkl")

        # if parallel_plot:
        #     self.lineplot_differences()
        # if plot_distributions:
        #     self.plot_distributions()
        print('Averages results of {} runs of HBAC'.format(num_runs))
        print('Average maximun negative biased cluster: ', mean(avg_bias), '±', np.std(avg_bias), ', min=', min(avg_bias))
        print('Average number of clusters: ', mean(avg_number_of_clusters), '±', np.std(avg_number_of_clusters))
        print('Average Sillhouette score: ', mean(avg_sillh), '±', np.std(avg_sillh), ', max=', max(avg_sillh))
        #return self.all_unscaled_combined

    def segment_predictor(self, runs=1, save_one_model=False, plot_roc_auc=False, shap_analysis=False):
        ### Get saved df
        data_xgb = pd.read_pickle("../all_unscaled_combined.pkl")
        data_xgb = data_xgb.drop(['predicted_value', 'true_value', 'errors'], axis=1)
        data_xgb = data_xgb.sample(frac=1)  # Shuffle data

        ############# Binary classification on clustering results
        x = data_xgb.drop('discriminated', axis=1)
        y = data_xgb['discriminated']
        X_xgb_train, X_xgb_test, y_xgb_train, y_xgb_test = train_test_split(x.to_numpy(), y.to_numpy(), test_size=0.3)

        # X_xgb_train, X_xgb_val, y_xgb_train, y_xgb_val = train_test_split(X_xgb_train, y_xgb_train,test_size=0.1)
        dtrain = xgb.DMatrix(X_xgb_train, label=y_xgb_train)
        dtest = xgb.DMatrix(X_xgb_test, label=y_xgb_test)

        ############# Binary classification on clustering results
        avg_acc = []
        avg_f1 = []
        avg_auc = []
        for i in range(runs):
            x = data_xgb.drop('discriminated', axis=1)
            y = data_xgb['discriminated']
            X_xgb_train, X_xgb_test, y_xgb_train, y_xgb_test = train_test_split(x.to_numpy(), y.to_numpy(),                                                                 test_size=0.3)
            dtrain = xgb.DMatrix(X_xgb_train, label=y_xgb_train)
            dtest = xgb.DMatrix(X_xgb_test, label=y_xgb_test)
            param = {'eta': 1, 'colsample_by_level': 0.5, 'objective': 'binary:logistic'}
            param['eval_metric'] = 'auc'
            evallist = [(dtest, 'eval'), (dtrain, 'train')]
            num_round = 50
            bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=5)
            ypred = bst.predict(dtest)
            bin_predictions = np.around(ypred, decimals=0)
            f1 = metrics.f1_score(y_xgb_test, bin_predictions, average='binary')
            acc = metrics.accuracy_score(y_xgb_test, bin_predictions)
            fpr, tpr, thresholds = metrics.roc_curve(y_xgb_test, bin_predictions)  # , pos_label=1)
            auc = metrics.auc(fpr, tpr)
            avg_acc.append(acc)
            avg_f1.append(f1)
            avg_auc.append(auc)
        print('Mean of {} runs'.format(runs))
        print('Accuracy of {}'.format(round(mean(avg_acc), 3)))
        print('F1 score of {}'.format(round(mean(avg_f1), 3)))
        print('AUC of {}'.format(round(mean(avg_auc), 3)))

        # Save last model
        if save_one_model:
            pkl.dump(bst, open('../segment_predictor.pkl', 'wb'))
        if plot_roc_auc:
            fpr, tpr, threshold = metrics.roc_curve(y_xgb_test, bin_predictions)
            roc_auc = metrics.auc(fpr, tpr)
            plt.title('ROC')
            plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
            plt.legend(loc='lower right')
            plt.plot([0, 1], [0, 1], 'r--')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.show()

        if shap_analysis:
            shap.initjs()
            explainer = shap.TreeExplainer(bst)
            shap_values = explainer.shap_values(x)
            pkl.dump(shap_values, open('../shap_values.pkl', 'wb'))
            shap.summary_plot(shap_values, features=x, feature_names=x.columns)
            shap.summary_plot(shap_values, features=x, feature_names=x.columns, plot_type='bar')


    def MICE(self, df, save_completed_df=False, plot_imputed_distributions=False):
        ### Using miceforest for multiple imputation https://pypi.org/project/miceforest/
        import miceforest as mf
        # Create kernel.
        kds = mf.ImputationKernel(
            df,
            datasets=3,
            mean_match_candidates=5,
            save_all_iterations=True,
            random_state=66,
        )
        # Run the MICE algorithm for 3 iterations on each of the datasets
        kds.mice(2)
        completed_dataset = kds.complete_data(dataset=0, inplace=False)
        ### Save completed dataset
        if save_completed_df:
            pd.to_pickle(completed_dataset, "completed_dataset.pkl")
        if plot_imputed_distributions:
            feature_with_missing_values = df.columns[df.isnull().any()]
            plot_features = [x for x in self.selected_features if x in feature_with_missing_values]
            kds.plot_imputed_distributions(variables=plot_features[:], iteration=2, wspace=0.5, hspace=0.8)
        return completed_dataset

    def get_max_bias_cluster(self, print_results = False, function=bias_acc):
        ''' This function returns the cluster linked to the highest negative bias of the newly introduced clusters
        fulldata (DataFrame) should include a column new_clusters '''
        # max_abs_bias = 100 changed to max_abs_bias
        fulldata = self.clustered_data #.drop('new_clusters',axis=1,inplace=True)
        max_neg_bias = 100
        best_cluster = -2
        for cluster_number in fulldata['clusters'].unique():
            if cluster_number == -1:  # Outliers in DBScan, which are excluded
                continue
            current_bias = (function(fulldata, cluster_number, "clusters"))  # abs function to find the highest bias
            print(f"{cluster_number} has bias {current_bias}") if print_results else ''
            if current_bias < max_neg_bias:
                max_neg_bias = current_bias
                best_cluster = cluster_number
        print('cluster with the highest discriminating bias:', best_cluster) if print_results else ''
        return best_cluster, max_neg_bias

    def pca_plot(self,  title='HBAC-KMeans', alpha=0.6):
        """ Function to perform dimensionality reduction on the features, so that we can create 2-dimensional scatterplots.
        Takes as input the entire dataset, selects the features on which we want to cluster, and stores them in a temporary pd Dataframe.
        This df is used to create a seaborn scatterplot. """

        data = self.clustered_data

        pca_features = data.drop(['scaled_errors', 'predicted_value', 'true_value', 'errors', 'clusters', 'new_clusters'],
                                 axis=1)
        other_features = data[['scaled_errors', 'predicted_value', 'true_value', 'errors', 'clusters', 'new_clusters']]
        # pca_features = data.drop(['predicted_value', 'true_value', 'errors', 'clusters', 'new_clusters'], axis=1)
        # other_features = data[['predicted_value', 'true_value', 'errors', 'clusters', 'new_clusters']]

        df = pd.DataFrame(pca_features)
        # print(PCA(n_components=2).fit(df).explained_variance_ratio_) if print_mode else ''

        pca = pd.DataFrame(PCA(n_components=2).fit_transform(df), index=df.index)

        # pca = PCA(n_components=2, svd_solver='full')
        # pca.fit(X)
        # PCA(n_components=2, svd_solver='full')
        # print(pca.explained_variance_ratio_)

        # print(PCA(n_components=2).fit_transform(df))
        temp_dataset = pca.join(other_features, how='left')
        temp_dataset.rename(columns={0: 'Principal Component 1'}, inplace=True)
        temp_dataset.rename(columns={1: 'Principal Component 2'}, inplace=True)

        scatterplot = sns.scatterplot(data=temp_dataset, x='Principal Component 1', y='Principal Component 2', alpha=alpha,
                                      hue="clusters", size='errors', sizes=(1, 200), palette="tab10")
        scatterplot.set_title(title)
        scatterplot.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), ncol=1)
        # plt.savefig('germancredit_kmeans.pngermancredit_kmeansg', dpi=300) # for saving a plot
        plt.show()