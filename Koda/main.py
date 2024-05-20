import numpy as np
import generateMatrix as gm
import debug
import sys

# NOTES for later:
# Matrix of words: Every document is a column, every word is a row
# For optimized search, we use a hashmap to store the index of the word in the matrix

DOCUMENT_FOLDER = "documents"
k = 10
cosinus_threadhold = 0.4

def matrix_to_file(matrix, file_name):
    # For debugging purposes, write the matrix to a file
    f = open(file_name, 'w' )
    f.write(np.array2string(matrix, threshold=np.inf, max_line_width=np.inf))
    f.close()

def svd(matrix, k):
    U, s, V = np.linalg.svd(matrix, full_matrices=False) # V is already transposed
    u_k = U[:, :k]
    s_k = np.diag(s[:k])
    v_k = V[:k, :]
    # matrix_k = np.dot(np.dot(U_k, s_k), V_k)
    return u_k, s_k, v_k

def build_query_vector(prompt, word_map, word_list):
    # Create a query vector from the prompt
    query_vector = np.zeros(len(word_list))
    words = prompt.split()
    for word in words:
        if word in word_map:
            query_vector[word_map[word]] += 1
    return query_vector

def s_k_inverse(s_k):
    # Invert the diagonal matrix s_k
    for i in range(s_k.shape[0]):
        if s_k[i, i] != 0:
            s_k[i, i] = 1 / s_k[i, i]
    return s_k

def cosine_similarity(v1, v2):
    # Find the angle between two vectors
    if not np.any(v1) or not np.any(v2):
        return 0
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def find_closest_documents(q_altered, v_k, file_names):
    # Find the closest documents to the query vector
    # q_altered is the query vector altered by the SVD
    # v_k is the V matrix from the SVD
    closest_documents = []
    for i in range(v_k.shape[1]):
        v = v_k[:, i]
        similarity = abs(cosine_similarity(q_altered, v))
        if similarity > cosinus_threadhold:
            closest_documents.append([i, similarity])
    
    # Sort the documents by similarity, highest similarity first
    closest_document_names = [[file_names[i],similarity] for i,similarity in closest_documents]
    closest_document_names = sorted(closest_document_names, key=lambda x: x[1])[::-1]
    
    return closest_document_names

def main():
    debug.set_debug_level(0)
    
    # Generate some data
    if len(sys.argv) <= 1:
        print("Please provide a prompt as an argument")
        return
    
    prompt = sys.argv[1]
    for i in range(2, len(sys.argv)):
        prompt += " " + sys.argv[i]
    debug.log("Prompt: " + prompt)
    
    file_names, word_map, word_list, matrix = gm.generate_matrix(DOCUMENT_FOLDER)
    matrix_to_file(matrix, "matrix.txt")
    u_k, s_k, v_k = svd(matrix, k)
    q = build_query_vector(prompt, word_map, word_list)
    q_altered = np.dot(np.dot(q.T, u_k), s_k_inverse(s_k))
    closest_documents = find_closest_documents(q_altered, v_k, file_names)
    print("The closest document to the prompt is: " + str(closest_documents))
    
if __name__ == "__main__":
    main()