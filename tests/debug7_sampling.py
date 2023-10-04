import os
import pandas as pd
from tqdm import tqdm


# FOLDER = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/projects/unicef/tmp/"
# FOLDER = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/scratch_folder/"
FOLDER = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/projects/unicef/school_selection/africa/extracted/"

FILTER_CLASS = "degurba"
# FILTER_CLASS = "grid_id"

schools = os.path.join(FOLDER, "africa_school_extraction.csv")

df = pd.read_csv(schools, encoding="latin-1")

# Calculate the proportions of each adm02 class
proportions = df[FILTER_CLASS].value_counts(normalize=True)

# Initialize a list to store the sampled DataFrames
sampled_dfs = []

# Initialize a dictionary to hold the DataFrames for each group
grouped_df = dict(tuple(df.groupby(FILTER_CLASS)))

# Total samples needed
total_samples_needed = 10000

sdf = pd.DataFrame()

# Iterate over the group proportions and sample data
for group, proportion in proportions.items():
    sampled_dfs.append(grouped_df[group].sample(1, replace=False))

# Concatenate all the sampled DataFrames
sdf = pd.concat(sampled_dfs, ignore_index=True)

for i in range(100):
    sampled_dfs = []

    # Iterate over the group proportions and sample data
    for group, proportion in proportions.items():
        sample_size = min(int(proportion * total_samples_needed), len(grouped_df[group]))
        sampled_dfs.append(grouped_df[group].sample(sample_size, replace=False))

    # Concatenate all the sampled DataFrames
    sampled_df = pd.concat(sampled_dfs, ignore_index=True)

    if i == 0:
        sdf = sampled_df
    else:
        # merge unique values
        sdf = pd.concat([sdf, sampled_df]).drop_duplicates().reset_index(drop=True)

    # If sdf has exactly 10,000 rows, break out of the loop
    if len(sdf) >= total_samples_needed:
        break

# Get the unique count of each grid_id in sdf
unique_count = sdf[FILTER_CLASS].value_counts()

# The number of features to remove to get to the total_samples_needed
features_to_remove = len(sdf) - total_samples_needed

# Progress bar for removal of features
for _ in tqdm(range(features_to_remove), total=features_to_remove):
    # Recompute the unique count of each grid_id
    current_count = sdf[FILTER_CLASS].value_counts()
    
    # Find the grid_id with the most features
    grid_id = current_count.idxmax()

    # If there are no features left for this grid_id, continue to the next iteration
    if unique_count[grid_id] <= 1:
        continue

    # Randomly select a row with the current grid_id and remove it from sdf
    random_select = sdf[sdf[FILTER_CLASS] == grid_id].sample()
    sdf = sdf.drop(index=random_select.index)

    # Update the unique_count for the removed grid_id
    unique_count[grid_id] -= 1

    # Update the features_to_remove count
    features_to_remove -= 1

sdf.to_csv(os.path.join(FOLDER, "africa_school_sampled.csv"), index=False, encoding="latin-1")

# Now sampled_df contains up to 10,000 stratified samples.
# import pdb; pdb.set_trace()

# top is africa
# bot is south-america
