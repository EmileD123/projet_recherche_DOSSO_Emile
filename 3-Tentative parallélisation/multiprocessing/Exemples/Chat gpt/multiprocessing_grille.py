import multiprocessing

#ATTENTION ERREUR DANS L'EXECUTION !

def calculate_row_sum(row_index, row, result_queue):
    row_sum = sum(row)
    result_queue.put((row_index, row_sum))

def parallel_matrix_sum(matrix, num_processes):
    result_queue = multiprocessing.Queue()
    processes = []

    # Nombre de lignes et de colonnes dans la matrice
    num_rows = len(matrix)
    num_cols = len(matrix[0])

    # Calcul du nombre de lignes par processus
    rows_per_process = num_rows // num_processes

    # Création des processus
    for i in range(num_processes):
        start_row = i * rows_per_process
        end_row = start_row + rows_per_process if i < num_processes - 1 else num_rows
        process = multiprocessing.Process(target=calculate_row_sum, args=(i, matrix[start_row:end_row], result_queue))
        processes.append(process)

    # Démarrage des processus
    for process in processes:
        process.start()

    # Attente de la fin des processus
    for process in processes:
        process.join()

    # Récupération des résultats partiels et calcul de la somme totale
    total_sum = 0
    results = {}
    while not result_queue.empty():
        row_index, row_sum = result_queue.get()
        results[row_index] = row_sum
        total_sum += row_sum

    return total_sum, results

if __name__ == "__main__":
    # Exemple de matrice 3x3
    example_matrix = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]

    num_processes = 2

    total_sum, row_sums = parallel_matrix_sum(example_matrix, num_processes)

    print(f"La somme totale de la matrice est {total_sum}.")
    print("Sommes partielles par ligne :", row_sums)
