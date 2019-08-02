import pandas as pd
import numpy as np


def find_feature_outliers(csv, out_csv, class_column, exclude=['fid', 'id', 'DN']):
    # Read csv
    df = pd.read_csv(csv)

    # Create z-score column
    df = df.assign(z_scr=pd.Series(np.zeros(len(df[class_column]))).values)

    # Get unique attributes in class column
    unique = df[class_column].unique().tolist()

    # Create exclude lust
    exclude.append(class_column)
    exclude.append('z_scr')
    exclude = list(set(exclude).intersection(df.columns.tolist()))

    z_scores = []

    # Calculate class values
    for csv_class in unique:
        sub_dataset = df.loc[df[class_column] == csv_class].drop(exclude, axis=1)

        # Calculate the medians
        medians = sub_dataset.median()

        # Calculate the median absolute deviation
        deviations = sub_dataset.subtract(medians)
        abs_deviations = deviations.abs()
        mad = abs_deviations.median()
        mad_std = mad.apply(lambda x: x * 1.4826)

        z_scores += sub_dataset.divide(mad_std).abs().mean(axis=1).tolist()

    df['z_scr'] = z_scores
    df.to_csv(out_csv)

    return out_csv


if __name__ == "__main__":
    in_csv = 'C:\\Users\\CFI\\Desktop\\satf_training\\training_data_no-z.csv'
    out_csv = 'C:\\Users\\CFI\\Desktop\\satf_training\\training_data_z.csv'
    find_feature_outliers(in_csv, out_csv, 'class')
