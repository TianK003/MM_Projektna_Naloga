import docx
import numpy as np
import os
import debug
from sklearn.datasets import fetch_20newsgroups
import sys
import re

matrix = None
testing = False
subjects = []

def add_new_document(A, U, S, V, a_new):
    """
    Adds a new document to the existing term-document matrix and updates the SVD components.
    
    Parameters:
    A (numpy.ndarray): Existing term-document matrix.
    U (numpy.ndarray): Existing left singular vectors.
    S (numpy.ndarray): Existing singular values (diagonal matrix).
    V (numpy.ndarray): Existing right singular vectors.
    a_new (numpy.ndarray): New document column vector.
    
    Returns:
    tuple: Updated (U, S, V) matrices.
    """
    # Compute the projection of a_new on the existing left singular vectors U
    p = np.dot(U.T, a_new)
    
    # Compute the residual vector r
    r = a_new - np.dot(U, p)
    
    # If r is approximately zero, a_new lies in the subspace spanned by U
    if np.linalg.norm(r) < 1e-10:
        # Form the augmented matrices
        U_k = U
        S_k = np.vstack([np.hstack([S, p.reshape(-1, 1)]), np.zeros((1, S.shape[1] + 1))])
        V_k = np.vstack([np.hstack([V, np.zeros((V.shape[0], 1))]), np.zeros((1, V.shape[1] + 1))])
        V_k[-1, -1] = 1
    else:
        # Normalize r to get the new singular vector component
        r = r / np.linalg.norm(r)
        
        # Form the augmented matrices
        U_k = np.hstack([U, r.reshape(-1, 1)])
        S_k = np.vstack([np.hstack([S, p.reshape(-1, 1)]), np.zeros((1, S.shape[1] + 1))])
        S_k[-1, -1] = np.linalg.norm(r)
        V_k = np.vstack([np.hstack([V, np.zeros((V.shape[0], 1))]), np.zeros((1, V.shape[1] + 1))])
        V_k[-1, -1] = 1
    
    return U_k, S_k, V_k

def add_new_documents():
    pass

def get_subject_from_document(data):
    split = data.split("\n")
    for j in range(0, len(split)):
        split_line = split[j].split()
        if len(split_line) == 0:
            print(data)
            print("No subject found - empty line")
            exit(0)
        if split_line[0] == "Subject:":
            subject_string = split[j].split("Subject:")[1].split("Re:")[-1].strip()
            return subject_string
    else:
        print(data)
        print("No subject found - no lines")
        print("Data:" + str(i))
        exit(0)

def get_subjects(data_limit):
    data = get_data("", data_limit)[1]
    
    debug.log("Getting subjects")
    for i in range(len(data)):
        subject_string = get_subject_from_document(data[i])
        subjects.append(subject_string)
    
    return subjects

def alter_data_for_testing(data):
    global testing
    global subjects
    debug.log("Altering data for testing")
    
    for i in range(len(data)):
        split = data[i].split("Lines:")
        for j in range(1, len(split)):
            data.append(split[j])

def read_docx_file(filename):
    doc = docx.Document(filename)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

def read_file(filename):
    return read_docx_file(filename)

def get_data(folder, data_limit):
    debug.log("Getting data")
    if folder == "":
        newsgroups_train = fetch_20newsgroups(subset='train')
        if data_limit>newsgroups_train.filenames.shape[0]:
            debug.log("No data limit")
            data_limit = newsgroups_train.filenames.shape[0]
        return newsgroups_train.filenames[:data_limit], newsgroups_train.data[:data_limit]
    
    file_names = os.listdir(folder)
    titles = []
    data = []
    for file_name in file_names:
        full_text = read_file(os.path.join(folder, file_name))
        data.append(full_text)
        titles.append(file_name)
    debug.log("Data gotten")
    return titles, data

def create_frequency_matrix(data): # data is a list of strings. Those strings are the content of the documents
    
    for i in range(len(data)):
        data[i] = data[i].lower()
        data[i] = re.sub('[\W_]+', ' ', data[i])
    
    debug.log("Creating frequency matrix")
    column_count = len(data)
    word_set = set() # Set of all words
    word_map = {} # Map of word to index
    
    # Create a set of all words (no duplicates)
    debug.log("Creating word set")
    for i in range(column_count):
        full_text = data[i]
        words = full_text.split()
        for word in words:
            word_set.add(word)
    
    # Create a list of all words and sort them
    word_list = list(word_set)
    # word_list.sort()
    
    # Create empty matrix with rows = number of words and columns = number of documents
    row_count = len(word_list)
    matrix = np.zeros((row_count, column_count))
    
    # Create a hashmap of word to index, so we can quickly find the index of a word
    for i in range(row_count):
        word_map[word_list[i]] = i
    
    # Fill the matrix with the count of each word in each document
    for j in range(column_count):
        full_text = data[j]
        words = full_text.split()
        for word in words:
            matrix[word_map[word], j] += 1
    
    debug.log("Matrix created")
    return word_map, word_list, matrix

def matrix_to_file(matrix, file_name):
    # For debugging purposes, write the matrix to a file
    f = open(file_name, 'w' )
    f.write(np.array2string(matrix, threshold=np.inf, max_line_width=np.inf))
    f.close()

def create_complex_matrix(matrix):
    debug.log("Creating complex matrix")
    
    n = matrix.shape[1] # Number of documents
    
    debug.log("Number of words: " + str(matrix.shape[0]))
    
    debug.log("Number of documents: " + str(n))
    
    debug.log("Generating percentage matrix")
    percentageMatrix = matrix/matrix.sum(axis=1)[:,None]
    debug.log("Percentage matrix generated")
    percentageMatrix = percentageMatrix * np.log(percentageMatrix)
    percentageMatrix = np.nan_to_num(percentageMatrix)
    debug.log("Percentage matrix log multiplied")
    percentageMatrix = percentageMatrix / np.log(n)
    debug.log("Percentage matrix divided by log(n)")
    percentageMatrix = percentageMatrix.sum(axis=1)
    debug.log("Percentage matrix summed")
    percentageMatrix = 1 + percentageMatrix # Matrika globalnih mer
    
    matrix = np.log(matrix + 1)
    matrix = matrix * percentageMatrix[:,None]
    
    debug.log("Complex matrix created")
    return matrix

def generate_matrix(folder = "", optimize = False, data_limit = 100000000):
    debug.log("Generating matrix")
    
    # file_names = os.listdir(folder)
    # Matrix of the number of times a word appears in each document
    titles, data = get_data(folder, data_limit)
    word_map, word_list, matrix = create_frequency_matrix(data)
    # Matrix based on point 4, local and global importance optimization. Uncomment for optimization
    if optimize:
        matrix = create_complex_matrix(matrix)
    return titles, word_map, word_list, matrix