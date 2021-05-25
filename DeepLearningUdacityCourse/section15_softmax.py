# Softmax's goal is to give catagories to different classes that
# are non negative. It accomplishes this by using exponents
# the weights will be between 0 and 1 and add up to 1

import numpy as np

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    expL = np.exp(L)
    sum = 0
    for index in expL:
        sum += index
    
    ans = []
    
    for i in range(len(L)):
        ans.append(expL[i]/sum)
        
    return ans

# For an test run, the program uses L=[5,6,7]
print(softmax([5,6,7]))