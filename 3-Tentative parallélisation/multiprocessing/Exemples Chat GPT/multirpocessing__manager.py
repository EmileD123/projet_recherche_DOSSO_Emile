import multiprocessing

def worker_function(index, queue):
    for i in range(5):
        message = f"Worker {index}: Message {i}"
        queue.put(message)

if __name__ == "__main__":
    # Create a multiprocessing Manager
    manager = multiprocessing.Manager()

    # Create a shared Queue using the Manager
    shared_queue = manager.Queue()

    # Number of worker processes
    num_processes = 3

    # Create and start worker processes
    processes = []
    for i in range(num_processes):
        process = multiprocessing.Process(target=worker_function, args=(i, shared_queue))
        processes.append(process)
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()

    # Retrieve and print messages from the shared queue
    print("Messages from the shared queue:")
    while not shared_queue.empty():
        message = shared_queue.get()
        print(message)

    # Clean up the manager
    manager.shutdown()
