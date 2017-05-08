
def compute_similarity_transform(X, Y, compute_optimal_scale=False):
  """
  A port of MATLAB's `procrustes` function to Numpy.
  Adapted from http://stackoverflow.com/a/18927641/1884420

  Args
    X: array NxM of targets, with N number of points and M point dimensionality
    Y: array NxM of inputs
    compute_optimal_scale: whether we compute optimal scale or force it to be 1

  Returns:
    d: squared error after transformation
    Z: transformed Y
    T: computed rotation
    b: scaling
    c: translation
  """
  import numpy as np

  muX = X.mean(0)
  muY = Y.mean(0)

  X0 = X - muX
  Y0 = Y - muY

  ssX = (X0**2.).sum()
  ssY = (Y0**2.).sum()

  # centred Frobenius norm
  normX = np.sqrt(ssX)
  normY = np.sqrt(ssY)

  # scale to equal (unit) norm
  X0 = X0 / normX
  Y0 = Y0 / normY

  # optimum rotation matrix of Y
  A = np.dot(X0.T, Y0)
  U,s,Vt = np.linalg.svd(A,full_matrices=False)
  V = Vt.T
  T = np.dot(V, U.T)

  # Make sure we have a rotation
  detT = np.linalg.det(T)
  V[:,-1] *= np.sign( detT )
  s[-1]   *= np.sign( detT )
  T = np.dot(V, U.T)

  traceTA = s.sum()

  if compute_optimal_scale:  # Compute optimum scaling of Y.
    b = traceTA * normX / normY
    d = 1 - traceTA**2
    Z = normX*traceTA*np.dot(Y0, T) + muX
  else:  # If no scaling allowed
    b = 1
    d = 1 + ssY/ssX - 2 * traceTA * normY / normX
    Z = normY*np.dot(Y0, T) + muX

  c = muX - b*np.dot(muY, T)

  return d, Z, T, b, c
