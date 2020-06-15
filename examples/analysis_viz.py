import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

base = "C:\\Users\\caspe\\Desktop\\Paper_2_StruturalDensity\\analysis\\"

data = pd.read_csv(base + "merged.csv")


# Sort?
sns.set(style="whitegrid")
sns.residplot(data["test"], data["res"], lowess=True, color="g")
plt.show()

# Calculate residuals
# Histogram quintiles(100), mean abs residual
