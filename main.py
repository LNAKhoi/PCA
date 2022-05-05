# from cv2 import mean
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import scipy.stats as stats
# import seaborn as sns
# from sklearn.preprocessing import scale
# from sklearn.decomposition import PCA


# # set NumPy options
# np.set_printoptions(suppress=True)
# pd.set_option('display.max_rows', 20)

# # read CSV file with pandas
# df= pd.read_csv('Dataset/TeslaStock.csv')
# # setup data columns
# df.columns=['Date','Open','High','Low','Close','Volume','Adj Close']

# #convert dates

# X = df.loc[:, "Open":]
# # dependent variable data
# y = df.Date


# # means and variances per group
# def print_mean_and_sd_by_group(variables, group_variable):
#     data_group_by = variables.groupby(group_variable)

#     print("## Means:")
#     print(data_group_by.apply(np.mean))

#     print("\n## Standard deviations:")
#     print(data_group_by.apply(np.std))

#     print("\n## Sample sizes:")
#     print(pd.DataFrame(data_group_by.apply(len)))

# # standardising variables

# standardisedX= scale(X)
# standardisedX = pd.DataFrame(standardisedX, index=X.index, columns=X.columns)
# print(standardisedX.apply(np.mean))
# print(standardisedX.apply(np.std))

# # principal component analysis

# pca = PCA().fit(standardisedX)

# def pca_summary(pca, standardised_data, out=True):
#     names = ["PC"+str(i) for i in range(1, len(pca.explained_variance_ratio_)+1)]
#     a = list(np.std(pca.transform(standardised_data), axis=0))
#     b = list(pca.explained_variance_ratio_)
#     c = [np.sum(pca.explained_variance_ratio_[:i]) for i in range(1, len(pca.explained_variance_ratio_)+1)]
#     columns = pd.MultiIndex.from_tuples([("sdev", "Standard deviation"), ("varprop", "Proportion of Variance"), ("cumprop", "Cumulative Proportion")])
#     summary = pd.DataFrame(list(zip(a, b, c)), index=names, columns=columns)
#     if out:
#         print("Importance of components:")
#         print(summary)
#     return summary


# summary = pca_summary(pca, standardisedX)
# print(summary.sdev)
# print(np.sum(summary.sdev**2))

# # how many principal components to retain
# def scree_plot(pca, standardised_values):
#     y = np.std(pca.transform(standardised_values), axis=0)**2
#     x = np.arange(len(y)) + 1
#     plt.plot(x, y, "o-")
#     plt.xticks(x, ["Comp."+str(i) for i in x], rotation=60)
#     plt.ylabel("Variance")
#     plt.show()


# plt.figure()
# scree_plot(pca, standardisedX)

# # loadings for the principal components
# print(pca.components_[0])
# print(np.sum(pca.components_[0]**2))


# def calc_pc(variables, loadings):
#     # find the number of samples in the data set and the number of variables
#     num_samples, num_variables = variables.shape
#     # make a vector to store the component
#     pc = np.zeros(num_samples)
#     # calculate the value of the component for each sample
#     for i in range(num_samples):
#         value_i = 0
#         for j in range(num_variables):
#             value_ij = variables.iloc[i, j]
#             loading_j = loadings[j]
#             value_i = value_i + (value_ij * loading_j)
#         pc[i] = value_i
#     return pc


# print(calc_pc(standardisedX, pca.components_[0]))
# print(pca.transform(standardisedX)[:, 0])

# print(pca.components_[1])
# print(np.sum(pca.components_[1]**2))


# # scatter plots of the principal components
# def pca_scatter(pca, standardised_values, classifs):
#     foo = pca.transform(standardised_values)
#     bar = pd.DataFrame(list(zip(foo[:, 0], foo[:, 1], classifs)), columns=["PC1", "PC2", "Class"])
#     sns.lmplot(x="PC1", y="PC2", data=bar, hue="Class", fit_reg=False)
#     plt.show()


# pca_scatter(pca, standardisedX, y)
# print_mean_and_sd_by_group(standardisedX, y)
# importing the libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA


# set NumPy options
data= pd.read_csv('Dataset/TeslaStock.csv')
# setup data columns
data.columns=['Date','Open','High','Low','Close','Volume','Adj Close']

#convert dates

X = data.loc[:, "Open":]
# dependent variable data
y = data.Date

print("## Data:")
print(data)

print("## Head:")
print(data.head())

print("## Tail:")
print(data.tail())

print("## Info:")
data.info()

# plotting multivariate data
pd.plotting.scatter_matrix(data.loc[:, "Open":"Adj Close"], diagonal="kde", figsize=(10, 5))
plt.show()

for i in range(2, 14):
     sns.lmplot(x=["V" + str(i)], y="V" + str(i + 1), data=data, hue="V1", fit_reg=False)

ax = data[['Date','Open','High','Low','Close','Volume','Adj Close']].plot(figsize=(10, 5))
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

ax = data[['Open','High','Low','Close','Volume','Adj Close']].plot(figsize=(10, 5))
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

ax = data[['High','Low','Close','Volume','Adj Close']].plot(figsize=(10, 5))
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

ax = data[['Low','Close','Volume','Adj Close']].plot(figsize=(10, 5))
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

ax = data[['Volume','Adj Close']].plot(figsize=(10, 5))
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# calculating summary statistics for multivariate data

print(X.apply(np.mean))
print(X.apply(np.std))
print(X.apply(np.max))
print(X.apply(np.min))



# calculating correlations for multivariate data
corr = stats.pearsonr(X.Open, X.High)
print("p-value:\t", corr[1])
print("cor:\t\t", corr[0])

corr_mat = X.corr()
print(corr_mat)

plt.figure()
sns.heatmap(corr_mat, vmax=1., square=False).xaxis.tick_top()
plt.show()

# Hinton diagram
def hinton(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()
    if not max_weight:
        max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))
    ax.patch.set_facecolor('lightgray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    for (x, y), w in np.ndenumerate(matrix):
        color = 'red' if w > 0 else 'blue'
        size = np.sqrt(np.abs(w))
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)
    n_ticks = matrix.shape[0]
    ax.xaxis.tick_top()
    ax.set_xticks(range(n_ticks))
    ax.set_xticklabels(list(matrix.columns), rotation=90)
    ax.set_yticks(range(n_ticks))
    ax.set_yticklabels(matrix.columns)
    ax.grid(False)
    ax.autoscale_view()
    ax.invert_yaxis()


plt.figure()
hinton(corr_mat)
plt.show()


def most_highly_correlated(my_dataframe, num_to_report):
    # find the correlations
    cor_matrix = my_dataframe.corr()
    # set the correlations on the diagonal or lower triangle to zero,
    # so they will not be reported as the highest ones:
    cor_matrix *= np.tri(*cor_matrix.values.shape, k=-1).T
    # find the top n correlations
    cor_matrix = cor_matrix.stack()
    cor_matrix = cor_matrix.reindex(cor_matrix.abs().sort_values(ascending=False).index).reset_index()
    # assign human-friendly names
    cor_matrix.columns = ["FirstVariable", "SecondVariable", "Correlation"]
    return cor_matrix.head(num_to_report)


print(most_highly_correlated(X, 10))


# standardising variables
standardisedX = scale(X)
standardisedX = pd.DataFrame(standardisedX, index=X.index, columns=X.columns)

print(standardisedX.apply(np.mean))
print(standardisedX.apply(np.std))


# principal component analysis
pca = PCA().fit(standardisedX)


def pca_summary(pca, standardised_data, out=True):
    names = ["PC"+str(i) for i in range(1, len(pca.explained_variance_ratio_)+1)]
    a = list(np.std(pca.transform(standardised_data), axis=0))
    b = list(pca.explained_variance_ratio_)
    c = [np.sum(pca.explained_variance_ratio_[:i]) for i in range(1, len(pca.explained_variance_ratio_)+1)]
    columns = pd.MultiIndex.from_tuples([("sdev", "Standard deviation"), ("varprop", "Proportion of Variance"), ("cumprop", "Cumulative Proportion")])
    summary = pd.DataFrame(list(zip(a, b, c)), index=names, columns=columns)
    if out:
        print("Importance of components:")
        print(summary)
    return summary


summary = pca_summary(pca, standardisedX)
print(summary.sdev)
print(np.sum(summary.sdev**2))


# how many principal components to retain
def scree_plot(pca, standardised_values):
    y = np.std(pca.transform(standardised_values), axis=0)**2
    x = np.arange(len(y)) + 1
    plt.plot(x, y, "o-")
    plt.xticks(x, ["Comp."+str(i) for i in x], rotation=60)
    plt.ylabel("Variance")
    plt.show()


plt.figure()
scree_plot(pca, standardisedX)

print(summary.sdev**2)

# loadings for the principal components
print(pca.components_[0])
print(np.sum(pca.components_[0]**2))


def calc_pc(variables, loadings):
    # find the number of samples in the data set and the number of variables
    num_samples, num_variables = variables.shape
    # make a vector to store the component
    pc = np.zeros(num_samples)
    # calculate the value of the component for each sample
    for i in range(num_samples):
        value_i = 0
        for j in range(num_variables):
            value_ij = variables.iloc[i, j]
            loading_j = loadings[j]
            value_i = value_i + (value_ij * loading_j)
        pc[i] = value_i
    return pc


print(calc_pc(standardisedX, pca.components_[0]))
print(pca.transform(standardisedX)[:, 0])

print(pca.components_[1])
print(np.sum(pca.components_[1]**2))


# scatter plots of the principal components
def pca_scatter(pca, standardised_values, classifs):
    foo = pca.transform(standardised_values)
    bar = pd.DataFrame(list(zip(foo[:, 0], foo[:, 1], classifs)), columns=["PC1", "PC2", "Class"])
    sns.lmplot(x="PC1", y="PC2", data=bar, hue="Class", fit_reg=False)
    plt.show()


pca_scatter(pca, standardisedX, y)

