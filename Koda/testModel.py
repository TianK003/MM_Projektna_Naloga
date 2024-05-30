import os
import subprocess

k_values = [10, 50, 100, 250, 500, 750, 1000]
threshold_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# threshold_values = [0.4, 0.7]
# k_values = [10, 50]
# threshold_values = [0.1, 0.2]

results = []

for k in k_values:
    for threshold in threshold_values:
        print(str(k) + ' = ' + str(threshold))
        call_command = ['python', 'main.py', '-t', '-l 1000']
        call_command.append('-k ' + str(k))
        call_command.append('-c ' + str(threshold))
        process = subprocess.run(call_command, capture_output=True, text=True)
        results.append(process.stdout[:-1])
        print(process.stdout[:-1])

formated_results = []
for i in range(len(k_values)):
    formated_results.append([])
    data_length = len(threshold_values)
    for j in range(data_length):
        formated_results[i].append(results[i * data_length + j])

print('Threshold:', end=' ')
for threshold in threshold_values:
    print(f'%7.2f' % threshold, end=' ')
print()

for i in range(len(k_values)):
    print(f'k=  %5s:' % k_values[i], end=' ')
    for j in range(len(formated_results[i])):
        print(f'  %5.5s' % formated_results[i][j], end=' ')
    print()
