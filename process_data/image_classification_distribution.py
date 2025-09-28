import pandas as pd
import glob
import os
import matplotlib.pyplot as plt

folder_path = "image_classification"
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

all_scores = []

for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    # Keep only rows where Amount Consumed is not null
    df = df[df["Calories"].notna()]
    if "Prediction Score" in df.columns:
        scores = df["Prediction Score"].dropna().tolist()
        all_scores.extend(scores)

# Calculate percentage above threshold
threshold = 0.6
above_threshold = [s for s in all_scores if s > threshold]
percent_above = len(above_threshold) / len(all_scores) * 100 if all_scores else 0

# Plot histogram
plt.figure(figsize=(10,6))
plt.hist(all_scores, bins=50, color='skyblue', edgecolor='black')
plt.axvline(threshold, color='red', linestyle='--', label=f'Score = {threshold}')
plt.xlabel("Prediction Score")
plt.ylabel("Frequency")
plt.title(f"Distribution of All Prediction Scores\n{percent_above:.2f}% above {threshold}")
plt.legend()
plt.show()
