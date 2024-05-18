import numpy as np
import generateMatrix as gm

# NOTES for later:
# Matrix of words: Every document is a column, every word is a row
# For optimized search, we use a hashmap to store the index of the word in the matrix

DOCUMENT_FOLDER = "documents"

def main():
    # Generate some data
    gm.generate_matrix(DOCUMENT_FOLDER)
    
if __name__ == "__main__":
    main()