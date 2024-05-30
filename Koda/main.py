import numpy as np
import generateMatrix as gm
import debug
import sys
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd
import argparse
import os
import shutil
import time

# NOTES for later:
# Matrix of words: Every document is a column, every word is a row
# For optimized search, we use a hashmap to store the index of the word in the matrix

document_folder = "documents"
k = 1000
cosinus_threadhold = 0.6
modes = ["unoptimized", "optimized"]
mode = modes[0]
data_limit = 1000000000

save_files_folder = os.path.join("savedMatricies", "savedMatricies_") # Appended in argsparse
u_file = "U.npy"
s_file = "s.npy"
v_file = "V.npy"
file_names_file = "file_names.npy"
word_map_file = "word_map.npy"
word_list_file = "word_list.npy"
matrix_file = "matrix.npy"
recompute = False
add_new_documents = False
testing = False

testing_score = 0 # For each correct document +1, for each document that was close but not correct + similarity score


def matrix_to_file(matrix, file_name):
    # For debugging purposes, write the matrix to a file
    f = open(file_name, 'w' )
    f.write(np.array2string(matrix, threshold=np.inf, max_line_width=np.inf))
    f.close()

def format_print_text(text, i):
    print_text = text
    max_size = 60
    terminal_size = os.get_terminal_size().columns-13
    if terminal_size > max_size:
        terminal_size = max_size
    if len(print_text) > terminal_size:
        print_text = print_text[:terminal_size-3] + "..."
    if len(print_text) < terminal_size:
        if (len(print_text) + 1)%2 == 0:
            print_text += " "
        space_symbol = " ."
        if i%2 == 0:
            space_symbol = "  "
        print_text += space_symbol*int((terminal_size-len(print_text))/2)
    
    return print_text

def compute_svd(matrix, k):
    debug.log("Performing SVD")
    U, s, V = randomized_svd(matrix, n_components=k)
    debug.log("SVD performed")
    
    return U, s, V

def svd(matrix, k, files_saved):
    
    if files_saved:
        debug.log("Loading SVD")
        U = np.load(os.path.join(save_files_folder, u_file))
        s = np.load(os.path.join(save_files_folder, s_file))
        V = np.load(os.path.join(save_files_folder, v_file))
        debug.log("SVD loaded")
    else:
        debug.log("Computing SVD")
        U, s, V = compute_svd(matrix, k)
        
        debug.log("Saving SVD")
        np.save(os.path.join(save_files_folder, u_file), U)
        np.save(os.path.join(save_files_folder, s_file), s)
        np.save(os.path.join(save_files_folder, v_file), V)
    
    if k > len(s):
        k = len(s)
    if k > data_limit:
        k = data_limit
    debug.log("k: " + str(k))
    u_k = U[:, :k]
    s_k = np.diag(s[:k])
    v_k = V[:k, :]
        
    return u_k, s_k, v_k

def build_query_vector(prompt, word_map, word_list):
    # Create a query vector from the prompt
    # debug.log("Building query vector")
    query_vector = np.zeros(len(word_list))
    words = prompt.split()
    for word in words:
        if word in word_map:
            query_vector[word_map[word]] += 1
    return query_vector

def s_k_inverse(s_k):
    # Invert the diagonal matrix s_k
    inverteds_k = np.zeros(s_k.shape)
    for i in range(s_k.shape[0]):
        if s_k[i, i] != 0:
            inverteds_k[i, i] = 1 / s_k[i, i]
    return inverteds_k

def cosine_similarity(v1, v2):
    # Find the angle between two vectors
    if not np.any(v1) or not np.any(v2):
        return 0
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def find_closest_documents(q_altered, v_k, file_names):
    # Find the closest documents to the query vector
    # q_altered is the query vector altered by the SVD
    # v_k is the V matrix from the SVD
    # debug.log("Finding closest documents")

    closest_documents = []
    for i in range(v_k.shape[1]):
        v = v_k[:, i]
        similarity = abs(cosine_similarity(q_altered, v))
        if similarity >= cosinus_threadhold:
            closest_documents.append([i, similarity])
    
    # Sort the documents by similarity, highest similarity first
    closest_document_names = [[file_names[i],similarity] for i,similarity in closest_documents]
    closest_document_names = sorted(closest_document_names, key=lambda x: x[1])[::-1]
    
    return closest_document_names

def testing_analysis(closest_documents, subject, temp_file_names, temp_data):
    global testing_score
    # debug.log("Testing analysis")
    for i in range(len(closest_documents)):
        if i >= 10:
            break
        index = np.nonzero(np.array(temp_file_names) == closest_documents[i][0])[0][0]
        current_data = temp_data[index]
        gotten_subject = gm.get_subject_from_document(current_data).lower()
        
        if gotten_subject == subject:
            if i == 0:
                testing_score += 1
                return
            
            testing_score += closest_documents[i][1] # The similarity score
            return
    
    debug.log("Failed to find subject: |" + subject + "|")

def analyze_results(closest_documents):
    # Print the results of the search
    
    if len(closest_documents) == 0:
        print("\033[94mNo documents found \033[0m")
        return
    
    while True:
        print("\n")
        print("\033[94mClosest documents: \033[0m")
        i = 0
        for document in closest_documents:
            print(f"\033[95m%4s: " % str(i),  end="\033[0m")
            print_text = format_print_text(document[0], i)
            print(f"%-s %.4f"% (print_text, document[1]))
            i+=1
        
        selected_document = input("\033[92mEnter the number of the document to view\033[0m (n/new for new prompt): ")
        if selected_document == "q" or selected_document == "quit" or selected_document == "exit":
            exit(0)
        if selected_document == "n" or selected_document == "new":
            break
        
        try:
            selected_document = int(selected_document)
            if selected_document < 0 or selected_document >= len(closest_documents):
                print("Invalid document number")
                continue
        except:
            print("Invalid document number")
            continue
        
        print(closest_documents[selected_document][0])
        temp_file_names, temp_data = gm.get_data(document_folder, data_limit)
        
        index = np.nonzero(np.array(temp_file_names) == closest_documents[selected_document][0])[0][0]
        print("-"*os.get_terminal_size().columns)
        print(temp_data[index])
        print("-"*os.get_terminal_size().columns)
        

def setup_parser():
    global document_folder
    global k
    global cosinus_threadhold
    global modes
    global mode
    global save_files_folder
    global recompute
    global data_limit
    global add_new_documents
    global testing
    parser = argparse.ArgumentParser(description='Find the closest document to a prompt')
    
    parser.add_argument('--folder', type=str, help='The folder containing the documents. Default is "' + document_folder + '"', default=document_folder)
    parser.add_argument('-o', '--online', help='Whether to use the online library of data.', action='store_true')
    parser.add_argument('-m', '--mode', type=str, help='The mode to run in. Options are ' + str(modes) + '. Default is ' + modes[0] + '.', default=modes[0])
    parser.add_argument('-k', '--k', type=int, help='The number of singular values to use. Default is ' + str(k) + '.', default=k)
    parser.add_argument('-c', '--cosine', type=float, help='The cosine similarity threshold. Default is ' + str(cosinus_threadhold) + '.', default=cosinus_threadhold)
    parser.add_argument('-d', '--debug', help='Print debug information.', action='store_true')
    parser.add_argument('--compute', help='Compute all files again.', action='store_true')
    parser.add_argument('-l', '--limit', type=int, help='The max number of documents to use. Default is ' + str(data_limit) + '.', default=data_limit)
    parser.add_argument('-a', '--add', help='Add all new documents to the database.', action='store_true')
    parser.add_argument('-t', '--test', help='Run the test suite. Always uses online data. Generates new tables.', action='store_true')
    args = parser.parse_args()
    
    is_online = args.online
    if args.test:
        is_online = True
    if is_online:
        document_folder = ""
    if args.folder and not is_online:
        document_folder = args.folder
    if args.k:
        k = args.k
    if args.cosine:
        cosinus_threadhold = args.cosine
    if args.mode:
        if args.mode not in modes:
            debug.log("Invalid mode. Options are " + str(modes))
            exit()
        mode = args.mode
    if args.debug:
        debug.set_debug_level(1)
    if args.compute:
        recompute = True
    if args.limit:
        data_limit = args.limit
    if args.add:
        add_new_documents = True
    if args.test:
        testing = True
        
    
    save_files_folder += mode + "_"
    save_files_folder += str(data_limit) + "_"
    if is_online:
        save_files_folder += "online"
    else:
        save_files_folder += document_folder
    if args.test:
        save_files_folder += "_test"
    
    
    debug.log("Folder: " + document_folder)
    debug.log("K: " + str(k))
    debug.log("Cosine: " + str(cosinus_threadhold))
    debug.log("Mode: " + args.mode)
    debug.log("Online: " + str(is_online))

def check_saved():
    global files_saved
    
    debug.log("Checking sava data")
    files_saved = True
    if not os.path.isdir(save_files_folder):
        files_saved = False
    if not os.path.exists(os.path.join(save_files_folder, u_file)):
        files_saved = False
    if not os.path.exists(os.path.join(save_files_folder, s_file)):
        files_saved = False
    if not os.path.exists(os.path.join(save_files_folder, v_file)):
        files_saved = False
    if not os.path.exists(os.path.join(save_files_folder, file_names_file)):
        files_saved = False
    if not os.path.exists(os.path.join(save_files_folder, word_map_file)):
        files_saved = False
    if not os.path.exists(os.path.join(save_files_folder, word_list_file)):
        files_saved = False
    if not os.path.exists(os.path.join(save_files_folder, matrix_file)):
        files_saved = False
    
    if recompute:
        files_saved = False
    
    
    if not files_saved:
        debug.log("Data not saved")
        if os.path.isdir(save_files_folder):
            shutil.rmtree(save_files_folder)
        os.makedirs(save_files_folder)
    
        
    return files_saved

def run():
    global k
    global document_folder
    global testing
    
    files_saved = check_saved()
    file_names, word_map, word_list, matrix = None, None, None, None
    
    debug.log("Getting data")
    if not files_saved:
        optimize = False
        if mode == modes[1]:
            optimize = True
        
        debug.log("Computing data")
        file_names, word_map, word_list, matrix = gm.generate_matrix(document_folder, optimize, data_limit)
        np.save(os.path.join(save_files_folder, file_names_file), file_names)
        np.save(os.path.join(save_files_folder, word_map_file), word_map)
        np.save(os.path.join(save_files_folder, word_list_file), word_list)
        np.save(os.path.join(save_files_folder, matrix_file), matrix)
    else:
        debug.log("Reading saved data")
        file_names = np.load(os.path.join(save_files_folder, file_names_file), allow_pickle=True).tolist()
        word_map = np.load(os.path.join(save_files_folder, word_map_file), allow_pickle=True).item()
        word_list = np.load(os.path.join(save_files_folder, word_list_file), allow_pickle=True).tolist()
        matrix = np.load(os.path.join(save_files_folder, matrix_file))
    
    u_k, s_k, v_k = svd(matrix, k, files_saved)
    is_k = s_k_inverse(s_k)
    
    if add_new_documents:
        debug.log("Adding new documents")
        new_file_names, new_data = gm.get_new_data(document_folder, file_names, data_limit)
        for i in range(len(new_file_names)):
            # Add new document
            debug.log("Adding new document: " + new_file_names[i])
            new_document_vector = build_query_vector(new_data[i], word_map, word_list)
            altered_document_vector = np.dot(np.dot(new_document_vector.T, u_k), is_k)
            v_k = np.hstack((v_k, np.zeros((v_k.shape[0], 1), dtype=v_k.dtype)))
            v_k[:,-1] = altered_document_vector
            file_names.append(new_file_names[i])
            
            # Add new word???
            data_split = new_data[i].split()
            new_words = set()
            for word in data_split:
                if word not in word_map:
                    new_words.add(word)
            
            for word in new_words:
                word_map[word] = len(word_list)
                word_list.append(word)
                count = 0
                for j in range(len(data_split)):
                    if word == data_split[j]:
                        count += 1
                word_query_vector = np.zeros(len(file_names))
                word_query_vector[-1] = count
                altered_word_query_vector = np.dot(np.dot(word_query_vector, v_k.T), s_k)
                u_k = np.vstack((u_k, np.zeros((1, u_k.shape[1]), dtype=u_k.dtype)))
                u_k[-1, :] = altered_word_query_vector
            

    if testing:
        subjects = gm.get_subjects(data_limit)
        temp_file_names, temp_data = gm.get_data(document_folder, data_limit)
        
        i = -1
        print("Testing")
        for subject in subjects:
            i+=1
            subject = subject.lower()
            debug.progress(i, len(subjects), without_debug=True)
            q = build_query_vector(subject, word_map, word_list)
            q_altered = np.dot(np.dot(q.T, u_k), is_k)
            closest_documents = find_closest_documents(q_altered, v_k, file_names)
            testing_analysis(closest_documents, subject, temp_file_names, temp_data)
        print("Score: " + str(testing_score))
        # print(testing_score)
    
    while not testing:
        prompt = input("\033[92mEnter a prompt\033[0m (q/quit/exit to quit): ").lower().strip()
        if prompt == "q" or prompt == "quit" or prompt == "exit":
            break
        q = build_query_vector(prompt, word_map, word_list)
        print(q.shape)
        print(u_k.shape)
        q_altered = np.dot(np.dot(q.T, u_k), is_k)
        closest_documents = find_closest_documents(q_altered, v_k, file_names)
        analyze_results(closest_documents)

def main():
    setup_parser()
    
    run()
    
if __name__ == "__main__":
    main()