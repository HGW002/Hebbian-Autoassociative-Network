# Hebbian-Autoassociative-Network
from Hagan's "Neural Network Design" textbook, implementing Hebbian learning and the pseudoinverse rule for an autoassociative memory network.


The problem focuses on designing and testing an **autoassociative memory network** using:
- **Hebbian Learning Rule**
- **Pseudoinverse Learning Rule**

Problem Summary
Given two 4-pixel binary prototype patterns \( p_1 \) and \( p_2 \), and a test pattern \( p_t \), we aim to:

1. **Check Orthogonality** of the input patterns.
2. **Design a Hebbian Autoassociator** using the input patterns.
3. **Test the network’s recall ability** using a test pattern and evaluate if the network performs as expected.
4. **Compare performance** with the pseudoinverse rule.

Included Files
- `hebbian_autoassociator.py`: Implements Hebbian rule and tests the output.
- `pseudoinverse_autoassociator.py`: Implements the pseudoinverse learning method for comparison.
  
Key Features
- Calculates weight matrices using Hebbian and pseudoinverse rules.
- Applies a `hardlims` activation function to simulate neuron thresholding.
- Measures Hamming distance to determine which stored pattern the network recalls.
- Discusses orthogonality and interference effects.

References
M.T. Hagan, H.B. Demuth, and M.H. Beale, *Neural Network Design*
Chapter 7 – Autoassociation and Pattern Recognition.
