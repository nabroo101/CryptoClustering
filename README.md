
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

<br>
The above code explains the steps taken to standardize and preprocess the data from the CSV file, creating a new DataFrame with the scaled values. The "coin_id" column is copied from the original data, and it becomes the index for the `df_scaled_data` DataFrame. Optionally, the code displays a sample of the scaled data for inspection.

---


### Find the Best Value for k Using the Original Scaled DataFrame

In this code, I am performing K-means clustering with different values of 'k' (the number of clusters) and storing the corresponding inertia values in a list.

1. `k = list(range(1, 11))`: create a list `k` containing the numbers from 1 to 10. This will be used as the range of 'k' values for K-means clustering.

2. `inertia = []`: An empty list named `inertia` is initialized to store the inertia values for each K-means model.

3. For loop:
   - The for loop iterates over each value of 'k' in the list `k`.
   - Inside the loop, I perform the following steps for each 'k':

     a. `k_model = KMeans(n_clusters=i)`: I create a KMeans model with 'i' clusters, where 'i' is the current value of 'k' in the loop.

     b. `k_model.fit(df_scaled_data)`: The KMeans model is fitted to the scaled data in the `df_scaled_data` DataFrame. The algorithm attempts to cluster the data points into 'i' clusters based on their similarity.

     c. `inertia.append(k_model.inertia_)`: The inertia value of the KMeans model is computed and appended to the `inertia` list. The inertia is a measure of how tightly the data points are clustered around their respective centroids. A lower inertia generally indicates better clustering.

After the loop finishes, the `inertia` list will contain the inertia values corresponding to each 'k' value, which can be used to evaluate and visualize the optimal number of clusters for my data. now we will plot the inertia values against the number of clusters to identify the "elbow" point, where the inertia starts to level off. This "elbow" point often indicates the optimal number of clusters for your K-means clustering.


In this code, i am plotting a line chart using the HoloViews extension for pandas to visualize the inertia values computed during the Elbow method.

```python
df_elbow_data.hvplot.line(
    x="k",
    y="inertia",
    title="Elbow method"
)
```

- `df_elbow_data`: This is the DataFrame containing the inertia values for different values of "k" (number of clusters).

- `hvplot.line()`: This is the HoloViews function used to create a line chart. The data from `df_elbow_data` will be plotted as a line chart.

- `x="k"` and `y="inertia"`: These parameters specify the columns to use for the x-axis and y-axis of the line chart, respectively. "k" will be used as the x-axis (representing the number of clusters), and "inertia" will be used as the y-axis (representing the inertia values).

- `title="Elbow method"`: This parameter sets the title of the plot to "Elbow method".

The resulting line chart will visually represent the inertia values for different values of "k". It helps you identify the "elbow" point, which indicates the optimal number of clusters for K-means clustering. The "elbow" point is the value of "k" where the inertia starts to level off, suggesting a good balance between the number of clusters and the compactness of each cluster.

To display the inertia values directly without plotting, i used the following line:

```python
df_elbow_data["inertia"]
```

This will display the inertia values for each corresponding value of "k" in the `df_elbow_data` DataFrame.

#### Cluster Cryptocurrencies with K-means Using the Original Scaled Data


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

