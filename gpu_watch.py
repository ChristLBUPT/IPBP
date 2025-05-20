from gpustat import GPUStatCollection
from jsonargparse import ArgumentParser
from time import sleep, time

def get_gpu_status():
    return GPUStatCollection.new_query(debug=True)

def watch(interval: int = 0.5):
    while True:
        start_time = time()
        gpu_status = get_gpu_status()
        gpu_status.print_formatted(force_color=True)
        print(f"\r\033[{len(gpu_status) + 2}A", end='')
        duration = time() - start_time
        sleep(max(0, interval - duration))


if __name__ == '__main__':
    watch(0.25)
