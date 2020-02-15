import sys
import numpy as np
import pandas as pd
sys.path.append('../lib')

from feature_selection import variance_threshold_selector


in_csv = 'E:\\SATF\\phase_IV_urban-classification\\training_data\\urban_segmentation.csv'
out_csv = 'E:\\SATF\\phase_IV_urban-classification\\training_data\\urban_segmentation.csv'

# Read csv
df = pd.read_csv(in_csv)
df_subset = df.drop('DN', axis=1)

thresholded = variance_threshold_selector(df, threshold=0.5)
removed = list(set(df.columns) - set(thresholded.columns))

thresholded['DN'] = df['DN']

import pdb; pdb.set_trace()

thresholded.to_csv(out_csv)

