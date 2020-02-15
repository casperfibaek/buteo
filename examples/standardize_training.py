import pandas as pd
# from sklearn.preprocessing import StandardScaler

# dataset = pd.read_csv('C:\\Users\\CFI\\Desktop\\satf_training\\urban_segmentation.csv')
# dataset_noid = dataset.drop('DN', axis=1)

# scaler = StandardScaler()
# scaled = pd.DataFrame(scaler.fit_transform(dataset_noid.values), columns=dataset_noid.columns, index=dataset_noid.index)
# scaled['DN'] = dataset['DN']


# scaled.to_csv('C:\\Users\\CFI\\Desktop\\satf_training\\urban_segmentation_standardized.csv')

training_set = pd.read_csv('C:\\Users\\CFI\\Desktop\\satf_training\\training_data_feature_selected.csv')
training_set_filter = training_set.filter(['DN', 'class'])
standardized_dataset = pd.read_csv('C:\\Users\\CFI\\Desktop\\satf_training\\urban_segmentation_standardized.csv')

merged = training_set_filter.merge(standardized_dataset, on='DN')
merged = merged.drop('Unnamed: 0', axis=1)
merged.to_csv('C:\\Users\\CFI\\Desktop\\satf_training\\training_data_feature_selected_standardized.csv')

import pdb; pdb.set_trace()
