import numpy as np

def weighted_quantile(values, weights, quintile=0.5, sorted=False):
  assert (quintile >= 0 and quintile <= 1)
  if sorted is False:
    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]

  length = len(values)
  wq_sum = 0
  wq_cumsum = np.zeros(length)
  wq = np.zeros(length)

  for i, v in enumerate(weights):
    wq_sum += v
    if i != 0:
      wq_cumsum[i] += wq_cumsum[i - 1] + v
    else:
      wq_cumsum[i] += v

  assert(wq_sum != 0)

  for v in range(length):
    wq[v] = (wq_cumsum[v] - (quintile * weights[v])) / wq_sum
    if wq[v] == quintile:
      return values[v]
    if wq[v] > quintile:
      low = abs(wq[v - 1] - quintile)
      high = abs(wq[v] - quintile)
      weight = low / (low + high)
      return (values[v - 1]) * weight + values[v] * (1 - weight)

  return values[length - 1]

if __name__ == "__main__":
  val = np.array([1, 3, 1, 5, 7])
  weights = np.array([1, 1, 1, 1, 1.1])
  
  bob = weighted_quantile(val, weights, 0.5)
  print(bob)