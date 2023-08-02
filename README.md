
# Crypto Clustering Analysis

## Overview

This data analysis project aims to analyze cryptocurrency market data, perform K-Means clustering, and visualize the results using Principal Component Analysis (PCA). The project involves using Python libraries such as pandas, scikit-learn, HoloViews, and NumPy.

## Project Structure

The project's directory structure is as follows:

```
crypto_clustering/
|-- Resources/
|   |-- crypto_market_data.csv
|-- README.md
|-- crypto_clustering.ipynb
|-- .gitignore
```

- `Resources/crypto_market_data.csv`: The CSV file containing the cryptocurrency market data used for analysis.
- `crypto_clustering.ipynb`: Jupyter Notebook file containing the Python code for the data analysis and visualization.

## Data Analysis Steps

The data analysis follows these major steps:

1. Data Loading: The cryptocurrency market data is loaded into a Pandas DataFrame for further processing.

2. Data Preprocessing: Missing values are handled, and data is cleaned and prepared for analysis.

3. K-Means Clustering: The data is standardized using StandardScaler, and K-Means clustering is applied to group cryptocurrencies based on various price change percentages.

4. Elbow Curve Analysis: An Elbow Curve is plotted to determine the optimal number of clusters for K-Means.

5. Principal Component Analysis (PCA): PCA is used to reduce the dimensionality of the data and visualize the clustering results in a lower-dimensional space.

6. Visualization: The results are visualized using scatter plots, line charts, and other plots to gain insights into the clustered data.

## Dependencies

The project relies on the following Python libraries:

- pandas
- scikit-learn
- hvplot.pandas
- NumPy

## How to Use

1. Clone the repository:

   ```
   git clone https://github.com/nabroo101/CryptoClustering
   cd crypto_clustering
   ```

2. Install the required libraries (if not already installed):

   ```
   pip install pandas scikit-learn hvplot numpy
   ```

3. Run the Jupyter Notebook:

   ```
   jupyter notebook crypto_clustering.ipynb
   ```

   The notebook contains all the code for the data analysis and visualization.

## Conclusion

This data analysis project provides valuable insights into the cryptocurrency market using K-Means clustering and PCA. By following the steps in the Jupyter Notebook, users can reproduce the analysis, explore the clustering results, and gain a better understanding of the trends in the cryptocurrency market.

Feel free to contribute to the project or use the code as a basis for further research and analysis!

