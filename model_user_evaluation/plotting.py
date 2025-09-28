import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
feature_names, xs, ys = FeatureLabelReducer(self.df_dict).get_x_y_data()
xs = xs[:, 3:]

print(feature_names[3:], xs[0], ys[0])
# Create a DataFrame for easy plotting
data = pd.DataFrame(xs, columns=feature_names[3:])
data['Target'] = ys

# Set Seaborn style
sns.set(style="whitegrid")

# Plot each feature against the target
plt.figure(figsize=(15, 6))
for i, feature in enumerate(feature_names[3:]):
    plt.subplot(1, len(feature_names[3:]), i+1)
    sns.scatterplot(x=data[feature], y=data['Target'])
    plt.xlabel(feature)
    if i == 0:
        plt.ylabel('Target')
    plt.title(f'{feature} vs Target')

plt.tight_layout()
plt.show()
