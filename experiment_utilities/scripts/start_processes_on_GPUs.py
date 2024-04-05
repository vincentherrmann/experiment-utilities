# It is used to start multiple processes on multiple GPUs.
# Its main purpose is starting the agents of a wandb sweep. To run it, execute
# python path_to_this_script "command_to_run" number_of_processes ids_of_available_GPUs
# python ~/Projects/experiment-utilities/experiment_utilities/scripts/start_processes_on_GPUs.py "wandb agent vincentherrmann/some-project/5awmhojb" 12 0 1 2 3

import argparse
import subprocess

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', type=str, help='Command to run')
    parser.add_argument('n_processes', type=int, default=1, help='Number of processes to start')
    parser.add_argument('gpu_ids', type=int, nargs='+', default=[0], help='IDs of GPUs to use')

    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    n_processes = args.n_processes
    gpu_ids = args.gpu_ids
    command = args.command

    for i in range(n_processes):
        gpu_id = gpu_ids[i % len(gpu_ids)]
        command_to_run = f"CUDA_VISIBLE_DEVICES={gpu_id} {command}"
        print(f"executing command: {command_to_run}")
        # execute command in background thread
        subprocess.Popen(command_to_run, shell=True)

if __name__ == '__main__':
    main()
