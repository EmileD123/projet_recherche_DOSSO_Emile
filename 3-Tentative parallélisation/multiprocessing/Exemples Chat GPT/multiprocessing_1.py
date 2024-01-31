import multiprocessing

def worker_function(start, end, result_queue):
    partial_sum = 0
    for i in range(start, end):
        partial_sum += i
    result_queue.put(partial_sum)

def parallel_sum(start, end, num_processes):
    result_queue = multiprocessing.Queue()
    processes = []

    # Calcul de la plage pour chaque processus
    step = (end - start) // num_processes
    for i in range(num_processes):
        process_start = start + i * step
        process_end = process_start + step if i < num_processes - 1 else end
        process = multiprocessing.Process(target=worker_function, args=(process_start, process_end, result_queue))
        processes.append(process)

    # Lancer les processus
    for process in processes:
        process.start()

    # Attendre la fin de chaque processus
    for process in processes:
        process.join()

    # Récupérer les résultats partiels
    total_sum = 0
    while not result_queue.empty():
        total_sum += result_queue.get()

    return total_sum

if __name__ == "__main__":
    start_value = 1
    end_value = 101 #le programme prend en compte jusqu'à end_value - 1
    num_processes = 4

    total_sum = parallel_sum(start_value, end_value, num_processes)
    print(f"La somme des nombres de {start_value} à {end_value} est {total_sum}.")
