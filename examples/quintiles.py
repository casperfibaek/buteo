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
    wq[v] = (wq_cumsum[v] - (0.5 * weights[v])) / wq_sum
    if wq[v] == quintile:
      return wq[v]
    if wq[v] > quintile:
      a = (value[v], weights[v])
      b = (value[v - 1], weight[v - 1])
      
  
  weighted_quantiles = (np.cumsum(weights) - (0.5 * weights)) / np.sum(weights)

  # return np.interp(0.5, weighted_quantiles, values),
  return values, wq

if __name__ == "__main__":
  val = np.array([1, 3, 1, 5, 7])
  weights = np.array([1, 1, 1, 1, 2])
  
  bob = weighted_quantile(val, weights, 0.5)
  print(bob)