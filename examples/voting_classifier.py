import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from scipy.stats import mode
from joblib import load

compare = 'C:\\Users\\CFI\\Desktop\\satf_training\\training_data_feature_selected_standardized.csv'
in_csv = 'C:\\Users\\CFI\\Desktop\\satf_training\\urban_segmentation.csv'
in_csv_std = 'C:\\Users\\CFI\\Desktop\\satf_training\\urban_segmentation_standardized.csv'
out_csv = 'C:\\Users\\CFI\\Desktop\\satf_training\\urban_segmentation_pred_ensemble.csv'
in_data = pd.read_csv(in_csv)
in_data_std = pd.read_csv(in_csv_std)
com_data = pd.read_csv(compare, index_col=0)
rf_model = load('C:\\Users\\CFI\\Desktop\\satf_training\\models\\RF_classifier.model')
gb_model = load('C:\\Users\\CFI\\Desktop\\satf_training\\models\\GB_classifier.model')
mlp_model = load('C:\\Users\\CFI\\Desktop\\satf_training\\models\\MLP_classifier.model')


in_col = list(in_data.columns)
com_col = list(com_data.columns)
overlap = list(set(in_col).intersection(set(com_col)))

to_classify_std = in_data_std.filter(overlap)
to_classify_std = to_classify_std.reindex(com_data.columns, axis=1)
to_classify_std = to_classify_std.drop(['class', 'DN'], axis=1)

pred_rf = rf_model.predict(to_classify_std)
pred_rf_prop = rf_model.predict_proba(to_classify_std)

pred_gb = gb_model.predict(to_classify_std)
pred_gb_prop = gb_model.predict_proba(to_classify_std)

pred_mlp = mlp_model.predict(to_classify_std)
pred_mlp_prop = mlp_model.predict_proba(to_classify_std)

combined_classes = np.array([pred_rf, pred_gb, pred_mlp])
combined_proba = np.array([pred_rf_prop, pred_gb_prop, pred_mlp_prop])

prob_class = []
confidence = []
length = pred_rf.shape[0]

for i in range(length):
    highest_prob = 0
    highest_prob_index = 0

    for j in range(0, 3):
        max_prob = np.max(combined_proba[j][i])

        if max_prob > highest_prob:
            highest_prob = max_prob
            highest_prob_index = j

    prob_class.append(combined_classes[highest_prob_index][i])
    confidence.append(highest_prob)


in_data['rf_pred'] = pred_rf
in_data['gb_pred'] = pred_gb
in_data['mlp_pred'] = pred_mlp
in_data['ens_p_pred'] = np.array(prob_class)
in_data['confidence'] = np.array(confidence)

in_data.to_csv(out_csv)
