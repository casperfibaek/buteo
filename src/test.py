from enum import Enum

arr = ('min', 'max', 'iqr')


class StatTypes(Enum):
    min = 1
    max = 2
    count = 3
    range = 4
    mean = 5
    median = 6
    std = 7
    kurtosis = 8
    skew = 9
    npskew = 10
    skewratio = 11
    variation = 12
    q1 = 13
    q3 = 14
    iqr = 15
    mad = 16
    madstd = 17
    within3std = 18
    within3std_mad = 19


statsToCalc = [StatTypes[x].value for x in arr].sort()
statsToCalc.sort()

print(statsToCalc1)
