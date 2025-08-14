import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(42)
n = 400
X1 = np.random.normal(0, 1, n)
X2 = 0.8 * X1 + np.random.normal(0, 0.6, n)
X3 = np.random.normal(0, 1, n)
X4 = -0.5 * X3 + np.random.normal(0, 0.7, n)
df = pd.DataFrame({"X1": X1, "X2": X2, "X3": X3, "X4": X4})

g = sns.pairplot(df, corner=True, diag_kind="hist", plot_kws={"s": 12, "alpha": 0.6})
g.fig.suptitle('Pair Plot', y=1.02)
plt.show()
