# Using HBAC to detect biased data segments
- Hierarchical Bias-Aware Clustering (HBAC) on regression models.
- Input: a trained model and a model's test data.
- Output: analysis of biased/discriminated data segment according to HBAC:
  - Comparing distributions of discriminated and remaining data.
  - Segment predictor: trains a XGBoost binary classifier to evaluate distinguishability of discriminated and remaining data with descriptive features. 
  
![alt text](https://github.com/honkert/hbac_error_detection/github_workflow.drawio.png?raw=true)

```python
# Initialize HBAC 
hbac = HBAC_analyser()

# In this case, input includes model path, X data and Y data
hbac.hbac_on_model(model_path, X_test, y_test) 

hbac.pca_plot()
discrimated_cluster, bias =  hbac.get_max_bias_cluster(print_results=True)

# Displaying results in dataframes
hbac.clustered_data

# Mean per feature 'discrimnated' cluster vs 'remaining' clusters
hbac.mean_clusters

# Plot 3 most different features' distributions
hbac.plot_distributions(plot_top_features = 3)

# Train XGBoost a binary classifier to predict whther a datapoint will be discrimnated or not, without using error as feature.
hbac.segment_predictor(plot_roc_auc=True,shap_analysis=True)
```

Also see **example.ipynb**.

For the use of HBAC on classification models, see https://github.com/Sm2468/msc_thesis/tree/master/hbac%20scripts, on which this project was based.
