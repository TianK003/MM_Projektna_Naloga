from sklearn.datasets import fetch_20newsgroups
from pprint import pprint

import torch

# Check if CUDA is available and set the device
print("CUDA is available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else exit())

# Create a random matrix
A = torch.rand(1000, 1000, device=device)

# Perform SVD
U, S, VT = torch.svd(A)

# Move tensors back to CPU if needed
U = U.cpu()
S = S.cpu()
VT = VT.cpu()