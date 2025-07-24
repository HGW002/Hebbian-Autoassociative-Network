import numpy as np

# Hebbian learning implementation


p1 = np.array([[-1], [-1], [1], [1]])
p2 = np.array([[1], [1], [-1], [1]])
pt = np.array([[1], [1], [1], [1]]) #For Test


P = np.hstack((p1, p2)) # Combine input patterns into a matrix P

W = np.dot(P, P.T) # Calculate the Hebbian weight 


output = np.dot(W, pt)

# Apply the hardlims activation function
def hardlims(vector):
    return np.where(vector >= 0, 1, -1)

a = hardlims(output)

print("Weight matrix W:")
print(W)
print("\nOutput before activation (W * pt):")
print(output)
print("\nFinal output after hardlims activation:")
print(a)

trained_patterns = [p1, p2]
distances = [np.sum(np.abs(a - pattern)) for pattern in trained_patterns]

if min(distances) == 0:
    print("The output exactly matches one of the input patterns.")
else:
    print(f"The output is closer to pattern {np.argmin(distances) + 1} with a Hamming distance of {min(distances)}.")
