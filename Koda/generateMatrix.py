import docx
import numpy as np
import os

def read_docx_file(filename):
    doc = docx.Document(filename)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

def read_file(filename):
    return read_docx_file(filename)

def create_matrix(folder, file_names):
    column_count = len(file_names)
    word_set = set() # Set of all words
    word_map = {} # Map of word to index
    
    # Create a set of all words (no duplicates)
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
    
    
    return matrix

def generate_matrix(folder):
    file_names = os.listdir(folder)
    matrix = create_matrix(folder, file_names)
    print(matrix)
    
