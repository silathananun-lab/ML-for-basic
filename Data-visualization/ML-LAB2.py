import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("train.csv")

# --------Histogram----------
df['Age'].hist(bins=20,rwidth=0.85)
# x
plt.xlabel("Age")
# y
plt.ylabel("Frequency")
plt.show()


# -------correlation Heatmap----------
corr =df.corr(numeric_only="True")
sns.heatmap(corr,annot=True,cmap="coolwarm")
plt.show()

