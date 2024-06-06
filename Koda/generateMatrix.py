import docx
import numpy as np
import os
import debug
from sklearn.datasets import fetch_20newsgroups
import sys
import re

testing = False
subjects = []

def correct_data_for_testing(data):
    # Remove all non-alphabetic/number characters and make all characters lowercase
    for i in range(len(data)):
        data[i] = data[i].lower()
        data[i] = re.sub('[\W_]+', ' ', data[i])
    
    return data


def get_new_data(folder, prev_file_names, data_limit):
    all_file_names, all_data = get_data(folder, data_limit)
    
    new_data = []
    new_file_names = []
    
    for i in range(len(all_file_names)):
        if all_file_names[i] not in prev_file_names:
            new_data.append(all_data[i])
            new_file_names.append(all_file_names[i])
    
    correct_data_for_testing(new_data)
    
    return new_file_names, new_data

def get_subject_from_document(data):
    # When testing, we need the "subject" of the testing data. This is the string after "Subject:"
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

def get_data(folder, data_limit): # Get the data from the chosen files. If no folder is given, we assume that the online library data is wanted
    debug.log("Getting data")
    if folder == "": # Use the online files
        newsgroups_train = fetch_20newsgroups(subset='train')
        if data_limit>newsgroups_train.filenames.shape[0]:
            debug.log("No data limit")
            data_limit = newsgroups_train.filenames.shape[0]
        file_names = newsgroups_train.filenames[:data_limit]
        # for i in range(len(file_names)):
        #     file_names[i] = get_subject_from_document(newsgroups_train.data[i])
        return file_names, newsgroups_train.data[:data_limit]
    
    # Else, use the files in the given folder. This code can be expanded to include other file types
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
    # Here we create the matrix, which counts how many times each word appears in each document. Since we are only storing integers, we do not need the document titles, as the data is already stored in the same order as the titles
    correct_data_for_testing(data)
    
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
    # Here we transform our frequency matrix into the weighted version calculated with given formulas
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