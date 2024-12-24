import numpy as np
import sklearn
from scipy.linalg import khatri_rao
from sklearn.svm import LinearSVC

# Loading train and test challenges and separating responses from it

train_data = np.loadtxt("train.dat")
test_data = np.loadtxt("test.dat")
x_train = train_data[:, :-1]
y_train = train_data[:, -1]
x_test = test_data[:, :-1]
y_test = test_data[:, -1]


def my_map(x):
 x_t = np.ones((x.shape[0],x.shape[1])) 
 for n in range(x.shape[0]):
  for i in range(x.shape[1]):
   for c in x[n,i:32]:
    x_t[n][i] = x_t[n][i] * (1-2*c)    # Computing x from c using this formula which was used in a typical arbiter PUF
 xxT = np.zeros((x_t.shape[0],x_t.shape[1],x_t.shape[1]))  
 xxTu = []
 xxtu = []
 for i in range(x_t.shape[0]):
    temp = x_t[i].reshape(32,1)
    xxT[i] = np.multiply(temp,temp.T)    # Computing x*x(transpose)
    for r in range(32):
      for c in range(r+1,32):
        xxtu.append(xxT[i][r][c])        # Appending the upper triangular terms of the x*x(transpose) matrix in a 1D array
    xxTu.append(xxtu)
    xxtu = []
 xxTu= np.array(xxTu)
 x_mapped = np.concatenate((xxTu,x_t),axis=1)    # Concatenating x with the flattened upper triangular terms of the matrix x*x(transpose) 
 return x_mapped


def my_fit(x_train,y_train):
  C_value = 11
  max_iter_value = 10000
  tol_value = 0.001
  model_SVC = LinearSVC(loss='hinge', tol=tol_value, C=C_value, max_iter=max_iter_value)   # Using Linear SVC model for classification
  x = my_map(x_train)      # Calling my_map to extract the mapped version of the training challenges
  y = y_train
  model_SVC.fit(x,y)
  w = model_SVC.coef_.flatten()    # Flattening the original weights extracted from the SVC model
  b = model_SVC.intercept_

  return w,b