import numpy as np

p1 = np.array([[-1], [-1], [1], [1]])
p2 = np.array([[1], [1], [-1], [1]])

P = np.hstack((p1, p2))

T = P

P_pseudo = np.linalg.pinv(P) # Calculate the pseudoinverse of P

W = np.dot(T, P_pseudo) # Compute the weight matrix using the pseudoinverse rule

pt = np.array([[1], [1], [1], [1]])

output = np.dot(W, pt)

def hardlims(vector):
    return np.where(vector >= 0, 1, -1)

a = hardlims(output)

print("Weight matrix W:")
print(W)
print("\nOutput before activation (W * pt):")
print(output)
print("\nFinal output after hardlims activation:")
print(a)


