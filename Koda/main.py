import numpy as np
import generateMatrix as gm

# NOTES for later:
# Matrix of words: Every document is a column, every word is a row
# For optimized search, we use a hashmap to store the index of the word in the matrix

DOCUMENT_FOLDER = "documents"
k = 2

def matrix_to_file(matrix, file_name):
    # For debugging purposes, write the matrix to a file
    f = open(file_name, 'w' )
    f.write(np.array2string(matrix, threshold=np.inf, max_line_width=np.inf))
    f.close()

def svd(matrix, k):
    U, s, V = np.linalg.svd(matrix, full_matrices=False) # V is already transposed
    U_k = U[:, :k]
    s_k = np.diag(s[:k])
    V_k = V[:k, :]
    matrix_k = np.dot(np.dot(U_k, s_k), V_k)
    return matrix_k

def main():
    # Generate some data
    matrix = gm.generate_matrix(DOCUMENT_FOLDER)
    matrix_k = svd(matrix, k)
    
    
if __name__ == "__main__":
    main()