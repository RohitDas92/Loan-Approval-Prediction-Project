#Code Block Starts

from scipy.stats import boxcox
transformed_data, lambda_value = boxcox(filtered_outliers_CLA['Current Loan Amount'])

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns


# Plot histogram
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.histplot(transformed_data, kde=True, color='skyblue')
plt.title('Histogram of Transformed Data')

# Q-Q plot
plt.subplot(1, 2, 2)
stats.probplot(transformed_data, dist="norm", plot=plt)
plt.title('Q-Q Plot')

plt.tight_layout()
plt.show()

# Shapiro-Wilk test
shapiro_stat, shapiro_p_value = stats.shapiro(transformed_data)
print(f'Shapiro-Wilk Test - Statistics: {shapiro_stat}, p-value: {shapiro_p_value}')

# Skewness and Kurtosis
skewness = stats.skew(transformed_data)
kurtosis = stats.kurtosis(transformed_data)
print(f'Skewness: {skewness}, Kurtosis: {kurtosis}')

# Density plot
plt.figure(figsize=(6, 4))
sns.kdeplot(transformed_data, color='skyblue')
plt.title('Density Plot of Transformed Data')
plt.show()

#Code Block Ends