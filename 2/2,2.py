import numpy as np
#skalary
sk=np.array(12)
#wektor
wk=np.array([12,3,6,14])
#macierz
mc=np.array([[12,3,6,14],
            [12,3,6,14],
            [12,3,6,14]])
print(mc.ndim)

z=np.zeros((200,30))
print(z.shape)
z=np.transpose(z)
print(z.shape)