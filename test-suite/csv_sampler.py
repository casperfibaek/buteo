from random import shuffle


def csv_sample_split(csv, sample_size=33, key=None):
    in_csv = open(csv, 'r').readlines()
    in_header = in_csv[0].strip().split(',')

    if key is not None:
        if key not in in_header:
            raise RuntimeError('Key not in csv') from None
    
    rows = list(range(len(in_csv) - 1))
    shuffle(rows)

    test_sample = rows[0:int(len(rows) * 0.33)]
    train_sample = rows[len(test_sample):]

    print(len(rows), len(test_sample), len(train_sample))


if __name__ == '__main__':
    csv = 'D:\\PhD\\Projects\\SavingsAtTheFrontiers\\pilot_analysis\\csv\\training_material_vector_01\\training_material_vector_01.csv'
    csv_sample_split(csv)
