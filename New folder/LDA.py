import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib as mpl
# read CSV file
df = pd.read_csv("data/pokemon.csv")


# avoid destruction of data, cast same types of pokemon into 1
df = df[df["type2"].isnull()].loc[
    :, ["sp_attack", "sp_defense", "attack", "defense", "speed", "hp", "type1"]
]
# initialize X and Y
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values


# Applying LDA with 2 components
lda = LinearDiscriminantAnalysis(n_components=2)
X_hat = lda.fit(X, y).fit_transform(X, y)

# getting colors from color maps in matplotlib
colors = mpl.cm.get_cmap(name="tab20").colors

# categorize the types of pokemon
categories = pd.Categorical(pd.Series(y)).categories

# create new dataframe with 2 components and types
newDf = pd.DataFrame(
    {"Component 1": X_hat[:, 0], "Component 2": X_hat[:, 1], "Type": pd.Categorical(pd.Series(y))}
)
print(newDf['Type'])
# merging all types into 1 scatter plot (nếu k có này thì biểu đồ sẽ bị phân ra thành nhiều thuộc tính riêng lẻ của pokemon)
fig, ax = plt.subplots(1, figsize=(10, 5))

for color, category in zip(colors, categories):
    (
        newDf.query("Type == @category")
        .plot.scatter(
            x="Component 1",
            y="Component 2",
            color=color,
            label=category,
            ax=ax,
            s=100,
            edgecolor="black"
        )
    )
plt.title('LDA')
plt.show()
