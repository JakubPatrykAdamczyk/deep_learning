import numpy as np
def naive_relu(x):
    assert len (x.shape)==2
    x=x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i,j]=max(x[i,i],0)
    return x

#liniowe dodawanie
def add_relu(x,y):
    assert len(x.shape)==2
    assert x.shape==y.shape
    x=x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i,j]+=y[i,j]
    return x

def native_add_matrix_and_vector(x,y):
    assert len(x.shape)==2  #matrix
    assert len(y.shape)==1  #vector
    assert x.shape[1]==y.shape[0]
    x=x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i,j]+=y[j]
    return x

def native_vector_dot(x,y):

    assert len(x.shape)==1  #vector
    assert len(y.shape)==1  #vector
    assert x.shape[0]==y.shape[0]
    z=0
    for i in range(x.shape[0]):
        z+=x[i]*y[1]
    return z

#iloczyn skalarny
def native_matrix__vectordot(x,y):
    assert len(x.shape)==2  #matrix
    assert len(y.shape)==1  #vector
    assert x.shape[1]==y.shape[0]
    z=np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            z[i]+=x[i,j]*y[j]
    return z

#iloczyn skalarny macierz
def native_matrix_dot(x,y):
    assert len(x.shape)==2  #matrix
    assert len(y.shape)==2  #matrix
    assert x.shape[1]==y.shape[0]
    
    z=np.zeros(x.shape[0],y.shape[1])
    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            row_x=x[i,:]
            column_y=y[:,j]
            z[i,j]=native_vector_dot(row_x,column_y)


x=np.random.random((64,3,32,10))
y=np.random.random((32,10))
z=np.max(x,y)

