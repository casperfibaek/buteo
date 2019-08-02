import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from joblib import load


compare_csv = 'C:\\Users\\CFI\\Desktop\\satf_training\\training_data_feature_selected.csv'
# compare_csv = 'E:\\SATF\\phase_IV_urban-classification\\training_data\\training_data_cleaned_phase_II.csv'
# in_csv = 'E:\\SATF\\phase_IV_urban-classification\\urban_segmentation.csv'
in_csv = 'C:\\Users\\CFI\\Desktop\\satf_training\\urban_segmentation.csv'
# out_csv = 'E:\\SATF\\phase_IV_urban-classification\\urban_segmentation_pred.csv'
out_csv = 'C:\\Users\\CFI\\Desktop\\satf_training\\urban_segmentation_pred.csv'
# model = 'E:\\SATF\\phase_IV_urban-classification\\training_data\\training_data_cleaned_phase_II.model'
model = 'C:\\Users\\CFI\\Desktop\\satf_training\\models\\training_data_feature_selection.model'

in_data = pd.read_csv(in_csv)
com_data = pd.read_csv(compare_csv, index_col=0)

in_col = list(in_data.columns)
com_col = list(com_data.columns)
overlap = list(set(in_col).intersection(set(com_col)))

to_classify = in_data.filter(overlap)
to_classify = to_classify.reindex(com_data.columns, axis=1)
to_classify = to_classify.drop(['class', 'DN'], axis=1)

# import pdb; pdb.set_trace()

rf_model = load(model)

pred = rf_model.predict(to_classify)
in_data['pred'] = pred

in_data.to_csv(out_csv)



# X = data.drop(['DN', 'class'], axis=1)
# y = data['class']