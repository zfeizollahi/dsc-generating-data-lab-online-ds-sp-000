# Generating Data - Lab

## Introduction

In this lab, we shall practice some of the data generation techniques that we saw in the previous lesson in order to generate datasets for regression and classification purposes. We will run a couple of simple simulations to help us generate different datasets by controlling noise and variance parameters in the data generation process. We will also look at the statistical indicators and visual output to see how these parameters affect the accuracy of an algorithm. 

## Objectives
In this lab you will:

- Generate datasets for classification problems
- Generate datasets for regression problems

## Generate data for classfication

Use `make_blobs()` to create a binary classification dataset with 100 samples, 2 features, and 2 centers (where each center corresponds to a different class label). Set `random_state = 42` for reproducibility.

_Hint: Here's a link to the documentation for_ [`make_blobs()`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html).


```python
import pandas as pd
```


```python
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
```


```python
# Your code here 
X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=42)
df = pd.DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
```


```python

df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
      <th>y</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-2.988372</td>
      <td>8.828627</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.722930</td>
      <td>3.026972</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-3.053580</td>
      <td>9.125209</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5.461939</td>
      <td>3.869963</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.867339</td>
      <td>3.280312</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Place the data in a `pandas` DataFrame called `df`, and inspect the first five rows of the data. 

_Hint: Your dataframe should have three columns in total, two for the features and one for the class label._ 


```python
# Your code here 
```

Create a scatter plot of the data, while color-coding the different classes.

_Hint: You may find this dictionary mapping class labels to colors useful: 
`colors = {0: 'red', 1: 'blue'}`_


```python
# Your code here 
colors = {0: 'red', 1: 'blue'}
fig, ax = plt.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', color=colors[key])
plt.show();
```


![png](index_files/index_10_0.png)


Repeat this exercise two times by setting `cluster_std = 0.5` and `cluster_std = 2`. 

Keep all other parameters passed to `make_blobs()` equal. 

That is:
* Create a classification dataset with 100 samples, 2 features, and 2 centers using `make_blobs()` 
    * Set `random_state = 42` for reproducibility, and pass the appropriate value for `cluster_std`  
* Place the data in a `pandas` DataFrame called `df`  
* Plot the values on a scatter plot, while color-coding the different classes 

What is the effect of changing `cluster_std` based on your plots? 


```python
# Your code here: 
# cluster_std = 0.5
X, y = make_blobs(n_samples=100, n_features=2, centers=2, cluster_std = 0.5, random_state=42)
df = pd.DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
colors = {0: 'red', 1: 'blue'}
fig, ax = plt.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', color=colors[key])
plt.show();
```


![png](index_files/index_12_0.png)



```python
# Your code here: 
# clusted_std = 2
X, y = make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=2, random_state=42)
df = pd.DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
colors = {0: 'red', 1: 'blue'}
fig, ax = plt.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', color=colors[key])
plt.show();
```


![png](index_files/index_13_0.png)


# Your comments here
Makes more dense clusters vs. spread out

## Generate data for regression

Create a function `reg_simulation()` to run a regression simulation creating a number of datasets with the `make_regression()` data generation function. Perform the following tasks:

* Create `reg_simulation()` with `n` (noise) and `random_state` as input parameters
    * Make a regression dataset (X, y) with 100 samples using a given noise value and random state
    * Plot the data as a scatter plot 
    * Calculate and plot a regression line on the plot and calculate $R^2$ (you can do this in `statsmodels` or `sklearn`)
    * Label the plot with the noise value and the calculated $R^2$ 
    
* Pass a fixed random state and values from `[10, 25, 40, 50, 100, 200]` as noise values iteratively to the function above 
* Inspect and comment on the output 


```python
X, y = make_regression(n_samples=100, n_features = 1, noise=n, random_state = random_state)
X.shape, y.shape
```




    ((100, 1), (100,))




```python
# Import necessary libraries
from sklearn.datasets.samples_generator import make_regression
from sklearn.linear_model import LinearRegression
import numpy as np
def reg_simulation(n, random_state):
    
    # Generate X and y
    X, y = make_regression(n_samples=100, n_features=1, noise=n, random_state = random_state)
    # Use X,y to draw a scatter plot
    plt.scatter(X, y, color='blue')
    # Fit a linear regression model to X , y and calculate r2
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    r2 = lin_reg.score(X,y)
    y_pred = lin_reg.predict(X)
    # label and plot the regression line
    plt.plot(X, y_pred, color='red')
    plt.title('Noise level = {}, r2 = {}'.format(n, r2))
    plt.show()
    pass


random_state = random_state = np.random.RandomState(42)

for n in [10, 25, 40, 50, 100, 200]:
    reg_simulation(n, random_state)
```


![png](index_files/index_17_0.png)



![png](index_files/index_17_1.png)



![png](index_files/index_17_2.png)



![png](index_files/index_17_3.png)



![png](index_files/index_17_4.png)



![png](index_files/index_17_5.png)


# Your comments here
r2 goes down as the data is more disperse since it's harder to have a linear regression line for data that isn't truly linear. The error will be much larger, and therefore a lower r2 score.

## Summary 

In this lesson, we learned how to generate random datasets for classification and regression problems. We ran simulations for this and fitted simple models to view the effect of random data parameters including noise level and standard deviation on the performance of parameters, visually as well as objectively. These skills will come in handy while testing model performance and robustness in the future. 
