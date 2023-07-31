
### Liberies and extensions used in this project:

```python

import pandas as pd
import hvplot.pandas
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
```

In this section, you're importing the required libraries for your data analysis and clustering tasks. 

- `pandas` is a powerful library for data manipulation and analysis.
- `hvplot.pandas` is an extension for pandas that enables interactive plotting using the Holoviews library.
- `KMeans` is an implementation of the K-means clustering algorithm, a popular unsupervised machine learning algorithm for clustering data points into groups.
- `PCA` is used for Principal Component Analysis, a dimensionality reduction technique.
- `StandardScaler` is used to standardize the features by removing the mean and scaling to unit variance. It's often essential for clustering algorithms.


### Importing CSV file
```python

df_market_data = pd.read_csv(
    "Resources/crypto_market_data.csv",
    index_col="coin_id")
```


### Preparing the data 

```python

df_market_data.columns

# Standardize the features using StandardScaler
scaler = StandardScaler()


df_scaled_data = StandardScaler().fit_transform(df_market_data[['price_change_percentage_24h', 'price_change_percentage_7d',
       'price_change_percentage_14d', 'price_change_percentage_30d',
       'price_change_percentage_60d', 'price_change_percentage_200d',
       'price_change_percentage_1y']])
df_scaled_data[:5]

```
This code snippet normalizes the selected columns from the `df_market_data` DataFrame using the `StandardScaler` module from scikit-learn.

 `df_scaled_data = StandardScaler().fit_transform(df_market_data[['price_change_percentage_24h', 'price_change_percentage_7d', ... 'price_change_percentage_1y']])`: In this part, a subset of columns is selected from the `df_market_data` DataFrame using double square brackets. The selected columns are `['price_change_percentage_24h', 'price_change_percentage_7d', ..., 'price_change_percentage_1y']`. These columns contain numerical data that needs to be normalized.

   - `StandardScaler()` creates an instance of the `StandardScaler` class, which is used for standardization. Standardization scales the data so that it has a mean of 0 and a standard deviation of 1.
   - `fit_transform()` is a method of the `StandardScaler` class. It computes the mean and standard deviation of the selected columns and then performs the transformation to standardize the data. The result of the transformation is stored in the new DataFrame `df_scaled_data`.


this code snippet standardizes a subset of columns in the `df_market_data` DataFrame, making it easier to work with the data, especially when performing machine learning algorithms or clustering where feature scaling is important for accurate results. The standardized data is stored in the `df_scaled_data` DataFrame, which can be used for further analysis or modeling.






#### Step 1: Standardizing the Data

The code begins by standardizing selected columns from the `df_market_data` DataFrame. It creates a new DataFrame `df_scaled_data` to store the scaled values.

```python
df_scaled_data = pd.DataFrame(
    scaled_data,
    columns=['price_change_percentage_24h', 'price_change_percentage_7d',
             'price_change_percentage_14d', 'price_change_percentage_30d',
             'price_change_percentage_60d', 'price_change_percentage_200d',
             'price_change_percentage_1y']
)
```

#### Step 2: Copying the Coin IDs

Next, the code copies the coin IDs from the original data and adds them as a new column named "coin_id" in the `df_scaled_data` DataFrame.

```python
df_scaled_data["coin_id"] = df_market_data.index
```

#### Step 3: Setting the Index

The code sets the "coin_id" column as the index for the `df_scaled_data` DataFrame.

```python
df_scaled_data = df_scaled_data.set_index("coin_id")
```

#### Step 4: Displaying Sample Data (Optional)

Finally, the code can display a sample of the scaled data using the `head()` method.

This provides an overview of the standardized data for further analysis or visualization.

---

The above code explains the steps taken to standardize and preprocess the data from the CSV file, creating a new DataFrame with the scaled values. The "coin_id" column is copied from the original data, and it becomes the index for the `df_scaled_data` DataFrame. Optionally, the code displays a sample of the scaled data for inspection.



---
---
---
---
---






.
.
.
.
.
.
.
.
.
.




















In this section, you are performing data preprocessing and feature scaling. 

- First, you define the list `features` to include the names of the columns that represent your input features. Replace the `'feature1', 'feature2', ...` with the actual names of your features.
- If your dataset has a target variable (i.e., a label or class that you want to predict or cluster by), you can specify its column name in the `target` variable. The code then separates the features (X) and the target variable (y) accordingly. If you don't have a target variable or you're performing unsupervised learning (clustering), you can skip this step.
- Next, you create an instance of `StandardScaler` and use it to scale the feature data (`X`) using the `fit_transform` method. Scaling the features is essential for many machine learning algorithms, including clustering, as it ensures that all features contribute equally to the analysis.

```python
# Perform Principal Component Analysis (PCA) for dimensionality reduction if needed
# Replace n_components with the desired number of principal components you want to keep
# If you don't want to use PCA, you can skip this step and keep the original feature data.
n_components = 2  # Change this value as per your requirement

pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)
```

If you have a large number of features and want to reduce dimensionality, you can use PCA. It's not always necessary, but it can be helpful for visualization and reducing computation time in high-dimensional datasets.

- Specify the number of principal components you want to keep in the `n_components` variable. This determines the number of dimensions in the reduced feature space.
- Then, create an instance of `PCA` and apply it to the scaled feature data (`X_scaled`) using the `fit_transform` method. This will give you the reduced feature data (`X_pca`) with the specified number of principal components.

```python
# Clustering using K-means
# Replace n_clusters with the desired number of clusters you want to create
n_clusters = 3  # Change this value as per your requirement

kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X_pca)  # Fit the KMeans model and get cluster labels
```

In this section, you're performing clustering using K-means on the reduced feature data obtained from PCA (`X_pca`). 

- Specify the number of clusters you want to create in the `n_clusters` variable. K-means will try to group the data points into this number of clusters.
- Create an instance of `KMeans` and fit it to the reduced feature data using the `fit_predict` method. This will assign each data point to a cluster, and you'll get an array of cluster labels stored in the `clusters` variable.

Now, you have performed clustering on your data. The variable `clusters` contains the cluster assignments for each data point. You can use this information for further analysis or visualization.

Keep in mind that your code assumes specific column names for features and the target variable. Make sure to replace those column names with the actual ones from your dataset. Additionally, you can modify the values of `n_components` and `n_clusters` based on your specific needs and domain knowledge.

I hope this helps you in documenting your code and understanding the different steps involved in your homework challenge. If you have any further questions or need additional assistance, feel free to ask! Good luck with your homework!

1. Rename the `Crypto_Clustering_starter_code.ipynb` file as `Crypto_Clustering.ipynb`.

2. Load the `crypto_market_data.csv` into a DataFrame.

3. Get the summary statistics and plot the data to see what the data looks like before proceeding.

#### Prepare the Data

* Use the `StandardScaler()` module from `scikit-learn` to normalize the data from the CSV file.

* Create a DataFrame with the scaled data and set the "coin_id" index from the original DataFrame as the index for the new DataFrame.

    * The first five rows of the scaled DataFrame should appear as follows:

        ![The first five rows of the scaled DataFrame](https://static.bc-edx.com/data/dl-1-2/m19/lms/img/scaled_DataFrame.png)

#### Find the Best Value for k Using the Original Scaled DataFrame

Use the elbow method to find the best value for `k` using the following steps:

* Create a list with the number of k values from 1 to 11.
* Create an empty list to store the inertia values.
* Create a `for` loop to compute the inertia with each possible value of `k`.
* Create a dictionary with the data to plot the elbow curve.
* Plot a line chart with all the inertia values computed with the different values of `k` to visually identify the optimal value for `k`.
* Answer the following question in your notebook: What is the best value for `k`?

#### Cluster Cryptocurrencies with K-means Using the Original Scaled Data

Use the following steps to cluster the cryptocurrencies for the best value for `k` on the original scaled data:

* Initialize the K-means model with the best value for `k`.
* Fit the K-means model using the original scaled DataFrame.
* Predict the clusters to group the cryptocurrencies using the original scaled DataFrame.
* Create a copy of the original data and add a new column with the predicted clusters.
* Create a scatter plot using hvPlot as follows:
    * Set the x-axis as "price_change_percentage_24h" and the y-axis as "price_change_percentage_7d".
    * Color the graph points with the labels found using K-means.
    * Add the "coin_id" column in the `hover_cols` parameter to identify the cryptocurrency represented by each data point.

#### Optimize Clusters with Principal Component Analysis

* Using the original scaled DataFrame, perform a PCA and reduce the features to three principal components.

* Retrieve the explained variance to determine how much information can be attributed to each principal component and then answer the following question in your notebook:
    * What is the total explained variance of the three principal components?

* Create a new DataFrame with the PCA data and set the "coin_id" index from the original DataFrame as the index for the new DataFrame.

    * The first five rows of the PCA DataFrame should appear as follows:

        ![The first five rows of the PCA DataFrame](https://static.bc-edx.com/data/dl-1-2/m19/lms/img/PCA_DataFrame.png)


#### Find the Best Value for k Using the PCA Data

Use the elbow method on the PCA data to find the best value for `k` using the following steps:

* Create a list with the number of k-values from 1 to 11.
* Create an empty list to store the inertia values.
* Create a `for` loop to compute the inertia with each possible value of `k`.
* Create a dictionary with the data to plot the Elbow curve.
* Plot a line chart with all the inertia values computed with the different values of `k` to visually identify the optimal value for `k`.
* Answer the following question in your notebook:
    * What is the best value for `k` when using the PCA data?
    * Does it differ from the best k value found using the original data?


#### Cluster Cryptocurrencies with K-means Using the PCA Data

Use the following steps to cluster the cryptocurrencies for the best value for `k` on the PCA data:

* Initialize the K-means model with the best value for `k`.
* Fit the K-means model using the PCA data.
* Predict the clusters to group the cryptocurrencies using the PCA data.
* Create a copy of the DataFrame with the PCA data and add a new column to store the predicted clusters.
* Create a scatter plot using hvPlot as follows:
    * Set the x-axis as "PC1" and the y-axis as "PC2".
    * Color the graph points with the labels found using K-means.
    * Add the "coin_id" column in the `hover_cols` parameter to identify the cryptocurrency represented by each data point.
* Answer the following question:
    * What is the impact of using fewer features to cluster the data using K-Means?

> **Rewind** Recall that you learned how to create composite plots in a previous module. If you need a refresher on how to create these plots, review that module. You can also check [Composing Plots](https://holoviz.org/tutorial/Composing_Plots.html) in the hvPlot documentation.

### Requirements

#### Find the Best Value for k by Using the Original Data (15 points)

To receive all points, you must:

* Code the elbow method algorithm to find the best value for k. Use a range from 1 to 11. (5 points)

* To visually identify the optimal value for k, plot a line chart of all the inertia values computed with the different values of k. (5 points)

* Answer the following question: What’s the best value for k? (5 points)

#### Cluster the Cryptocurrencies with K-Means by Using the Original Data (10 points)

To receive all points, you must:

* Initialize the K-means model with four clusters by using the best value for k. (1 point)

* Fit the K-means model by using the original data. (1 point)

* Predict the clusters for grouping the cryptocurrencies by using the original data. Review the resulting array of cluster values. (3 points)

* Create a copy of the original data, and then add a new column of the predicted clusters. (1 point)

* Using hvPlot, create a scatter plot by setting `x="price_change_percentage_24h"` and `y="price_change_percentage_7d"`. Color the graph points with the labels that you found by using K-means. Then add the crypto name to the `hover_cols` parameter to identify the cryptocurrency that each data point represents. (4 points)

#### Optimize the Clusters with Principal Component Analysis (10 points)

To receive all points, you must:

* Create a PCA model instance, and set `n_components=3`. (1 point)

* Use the PCA model to reduce the features to three principal components. Then review the first five rows of the DataFrame. (2 points)

* Get the explained variance to determine how much information can be attributed to each principal component. (2 points)

* Answer the following question: What’s the total explained variance of the three principal components? (3 points)

* Create a new DataFrame with the PCA data. Be sure to set the `coin_id` index from the original DataFrame as the index for the new DataFrame. Review the resulting DataFrame. (2 points)

#### Find the Best Value for k by Using the PCA Data (10 points)

To receive all points, you must:

* Code the elbow method algorithm, and use the PCA data to find the best value for k. Use a range from 1 to 11. (2 points)

* To visually identify the optimal value for k, plot a line chart of all the inertia values computed with the different values of k. (5 points)

* Answer the following questions: What’s the best value for k when using the PCA data? Does it differ from the best value for k that you found by using the original data? (3 points)

#### Cluster the Cryptocurrencies with K-means by Using the PCA Data (10 points)

To receive all points, you must:

* Initialize the K-means model with four clusters by using the best value for k. (1 point)

* Fit the K-means model by using the PCA data. (1 point)

* Predict the clusters for grouping the cryptocurrencies by using the PCA data. Review the resulting array of cluster values. (3 points)

* Create a copy of the DataFrame with the PCA data, and then add a new column to store the predicted clusters. (1 point)

* Using hvPlot, create a scatter plot by setting `x="PC1"` and `y="PC2"`. Color the graph points with the labels that you found by using K-means. Then add the crypto name to the `hover_cols` parameter to identify the cryptocurrency that each data point represents. (4 points)

#### Visualize and Compare the Results (15 points)

To receive all points, you must:

* Create a composite plot by using hvPlot and the plus sign (`+`) operator to compare the elbow curve that you created from the original data with the one that you created from the PCA data. (5 points)

* Create a composite plot by using hvPlot and the plus (`+`) operator to compare the cryptocurrency clusters that resulted from using the original data with those that resulted from the PCA data. (5 points)

* Answer the following question: Based on visually analyzing the cluster analysis results, what’s the impact of using fewer features to cluster the data by using K-means? (5 points)

#### Coding Conventions and Formatting (10 points)

To receive all points, you must:

* Place imports at the top of the file, just after any module comments and docstrings, and before module globals and constants. (3 points)

* Name functions and variables with lowercase characters, with words separated by underscores. (2 points)

* Follow DRY (Don't Repeat Yourself) principles, creating maintainable and reusable code. (3 points)

* Use concise logic and creative engineering where possible. (2 points)

#### Deployment and Submission (10 points)

To receive all points, you must:

* Submit a link to a GitHub repository that’s cloned to your local machine and that contains your files. (4 points)

* Use the command line to add your files to the repository. (3 points)

* Include appropriate commit messages in your files. (3 points)

#### Code Comments (10 points)

To receive all points, your code must:

* Be well commented with concise, relevant notes that other developers can understand. (10 points)

### Grading

This project will be evaluated against the requirements and assigned a grade according to the following table:

| Grade | Points |
| --- | --- |
| A (+/-) | 90+ |
| B (+/-) | 80&ndash;89 |
| C (+/-) | 70&ndash;79 |
| D (+/-) | 60&ndash;69 |
| F (+/-) | < 60 |

### Submission

Each student is required to submit the URL of your GitHub repository for grading.

> **Note:** Projects are requirements for graduation. While you are allowed to miss up to two Challenge assignments and still earn your certificate, projects cannot be skipped.


> **Important:** **It is your responsibility to include a note in the README section of your repo specifying code source and its location within your repo**. This applies if you have worked with a peer on an assignment, used code in which you did not author or create sourced from a forum such as Stack Overflow, or you received code outside curriculum content from support staff such as an Instructor, TA, Tutor, or Learning Assistant. This will provide visibility to grading staff of your circumstance in order to avoid flagging your work as plagiarized.
>
> If you are struggling with a Challenge or any aspect of the curriculum, please remember that there are student support services available for you:
>
> 1. Office hours facilitated by your TA(s)
>
> 2. Tutor sessions ([sign up](https://tinyurl.com/BootCampTutorTeam))
>
> 3. Ask the class Slack channel/get peer support
>
> 4. AskBCS Learning Assistants

### References

