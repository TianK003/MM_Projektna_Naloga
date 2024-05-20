import docx
import numpy as np
import os
import debug

def read_docx_file(filename):
    doc = docx.Document(filename)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

def read_file(filename):
    return read_docx_file(filename)

def create_frequency_matrix(folder, file_names):
    debug.log("Creating frequency matrix")
    column_count = len(file_names)
    word_set = set() # Set of all words
    word_map = {} # Map of word to index
    
    # Create a set of all words (no duplicates)
    debug.log("Creating word set")
    for file_name in file_names:
        full_text = read_file(os.path.join(folder, file_name))
        words = full_text.split()
        for word in words:
            word_set.add(word)
    
    # Create a list of all words and sort them
    word_list = list(word_set)
    word_list.sort()
    
    # Create empty matrix with rows = number of words and columns = number of documents
    row_count = len(word_list)
    matrix = np.zeros((row_count, column_count))
    
    # Create a hashmap of word to index, so we can quickly find the index of a word
    for i in range(row_count):
        word_map[word_list[i]] = i
    
    # Fill the matrix with the count of each word in each document
    for j in range(column_count):
        full_text = read_file(os.path.join(folder, file_names[j]))
        words = full_text.split()
        for word in words:
            matrix[word_map[word], j] += 1
    
    debug.log("Matrix created")
    return word_map, word_list, matrix

def create_complex_matrix(matrix):
    debug.log("Creating complex matrix")
    
    n = matrix.shape[1] # Number of documents
    
    for i in range(matrix.shape[0]): # Loop through all words
        global_frequency = np.sum(matrix[i, :]) # Global frequency of the word
        p_sum = 0
        for j in range(n): # Loop through all documents
            local_frequency = matrix[i, j]
            p_ij = local_frequency / global_frequency
            if p_ij == 0:
                continue
            p_sum += p_ij * np.log(p_ij) / np.log(n)
        
        p_sum = -p_sum
        g_i = 1 - p_sum
        
        for j in range(n):
            local_frequency = matrix[i, j]
            l_ij = np.log(local_frequency + 1)
            matrix[i, j] = l_ij * g_i
            
    debug.log("Complex matrix created")
    return matrix

def generate_matrix(folder):
    debug.log("Generating matrix")
    
    file_names = os.listdir(folder)
    # Matrix of the number of times a word appears in each document
    word_map, word_list, matrix = create_frequency_matrix(folder, file_names)
    # Matrix based on point 4, local and global importance optimization. Uncomment for optimization
    matrix = create_complex_matrix(matrix)
    return file_names, word_map, word_list, matrix