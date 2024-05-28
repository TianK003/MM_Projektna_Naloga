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


def matrix_to_file(matrix, file_name):
    # For debugging purposes, write the matrix to a file
    f = open(file_name, 'w' )
    f.write(np.array2string(matrix, threshold=np.inf, max_line_width=np.inf))
    f.close()

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
    debug.log("Vk shape: " + str(v_k.shape))
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
            print("\033[95m" + str(i) + ": ", end="\033[0m")
            print(document[0] + " " + str(document[1]))
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
    parser = argparse.ArgumentParser(description='Find the closest document to a prompt')
    
    parser.add_argument('--folder', type=str, help='The folder containing the documents. Default is "' + document_folder + '"', default=document_folder)
    parser.add_argument('-o', '--online', help='Whether to use the online library of data.', action='store_true')
    parser.add_argument('--mode', type=str, help='The mode to run in. Options are ' + str(modes) + '. Default is ' + modes[0] + '.', default=modes[0])
    parser.add_argument('-k', '--k', type=int, help='The number of singular values to use. Default is ' + str(k) + '.', default=k)
    parser.add_argument('--cosine', type=float, help='The cosine similarity threshold. Default is ' + str(cosinus_threadhold) + '.', default=cosinus_threadhold)
    parser.add_argument('-d', '--debug', help='Print debug information.', action='store_true')
    parser.add_argument('--compute', help='Compute all files again.', action='store_true')
    parser.add_argument('-l', '--limit', type=int, help='The max number of documents to use. Default is ' + str(data_limit) + '.', default=data_limit)
    parser.add_argument('-a', '--add', help='Add all new documents to the database.', action='store_true')
    args = parser.parse_args()
    
    
    if args.online:
        document_folder = ""
    if args.folder and not args.online:
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
    
    save_files_folder += mode + "_"
    save_files_folder += str(data_limit) + "_"
    if args.online:
        save_files_folder += "online"
    else:
        save_files_folder += document_folder
    
    
    debug.log("Folder: " + document_folder)
    debug.log("K: " + str(k))
    debug.log("Cosine: " + str(cosinus_threadhold))
    debug.log("Mode: " + args.mode)
    debug.log("Online: " + str(args.online))

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
    
    if add_new_documents:
        debug.log("Adding new documents")
        gm.add_new_documents(document_folder, file_names, word_map, word_list, matrix, save_files_folder)
    
    
    while True:
        prompt = input("\033[92mEnter a prompt\033[0m (q/quit/exit to quit): ")
        if prompt == "q" or prompt == "quit" or prompt == "exit":
            break
        q = build_query_vector(prompt, word_map, word_list)
        q_altered = np.dot(np.dot(q.T, u_k), s_k_inverse(s_k))
        closest_documents = find_closest_documents(q_altered, v_k, file_names)
        analyze_results(closest_documents)

def main():
    setup_parser()
    
    run()
    
if __name__ == "__main__":
    main()