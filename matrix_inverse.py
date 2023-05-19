import numpy as np

e = np.exp(1)

arr = np.array([[e**6/(1 + e**2), -e**(12/5), e**(14/5)/(1 + e**2)],
                [-e**(18/5) , 1 + e**4 , -e**(12/5)],
                [e**(26/5)/(1 + e**2) , -e**(18/5) , e**6/(1 + e**2)]])

scaling = e**(9/100)/(e**2 - 1)
print(np.linalg.pinv(arr*scaling))


coil_sense = np.array(([[-e**(18/5) , 1 + e**4 , -e**(12/5)]]))
coil_weights = np.linalg.pinv(coil_sense)
print(np.linalg.pinv(coil_sense))

SNR = 