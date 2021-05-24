# The goal is to find the entropy of a given set using a list of actual results (Y) and the list of probabilities (P)
import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.

# formula: Cross-Entropy = -sum(yln(p) + (1-p)ln(1-p))
def cross_entropy(Y, P):
    ans = 0
    for i in range(len(Y)):
        ans-= Y[i] * np.log(P[i]) + (1-Y[i]) * np.log(1-P[i])
    return ans

# This is the example given
Y = [1,0,1,1]
P = [.4, .6, .1, .5]

# expected output is 4.828313737
print(cross_entropy(Y, P))