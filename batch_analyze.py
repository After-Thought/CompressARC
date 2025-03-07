import argparse
import subprocess
from typing import List, Tuple
import os
from multiprocessing import Pool
import multiprocessing
import yaml

def read_task_list(filename: str) -> List[Tuple[str, str]]:
    """Read a file containing split and task pairs, one per line.
    Expected format: split,taskid
    Example: training,272f95fa"""
    tasks = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # Skip empty lines and comments
                try:
                    split, task_id = line.split(',')
                    split = split.strip().lower()  # Normalize split name
                    task_id = task_id.strip()
                    # Validate split name
                    if split not in ['training', 'evaluation', 'test']:
                        print(f"Warning: Invalid split '{split}' for task {task_id}. Must be 'training', 'evaluation', or 'test'")
                        continue
                    tasks.append((split, task_id))
                except ValueError:
                    print(f"Warning: Invalid line format: '{line}'. Expected format: split,taskid")
                    continue
    return tasks

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def process_task(args) -> None:
    """Process a single task - used by the process pool"""
    split, task_id, analyzer_script, config_path = args
    print(f"\nProcessing task {task_id} from {split} split...")
    try:
        subprocess.run(['python', analyzer_script, 
                      '--split', split, 
                      '--task', task_id,
                      '--config', config_path], 
                     check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error processing task {task_id}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Batch process multiple ARC tasks')
    parser.add_argument('--input-file', type=str,
                      help='Path to file containing split,taskid pairs')
    parser.add_argument('--config', type=str,
                      help='Path to YAML config file')
    parser.add_argument('--analyzer-script', type=str, default='analyze_example.py',
                      help='Path to the analyzer script (default: analyze_example.py)')
    parser.add_argument('--num-processes', type=int, default=multiprocessing.cpu_count(),
                      help='Number of parallel processes to use (default: number of CPU cores)')
    args = parser.parse_args()

    if args.config:
        # Load tasks from config file
        config = load_config(args.config)
        split = config['cli_args']['split']
        tasks = [(split, task_id) for task_id in config['cli_args']['tasks']]
    elif args.input_file:
        # Read tasks from input file
        tasks = read_task_list(args.input_file)
    else:
        parser.error("Either --input-file or --config must be specified")
    
    # Prepare arguments for parallel processing
    process_args = [(split, task_id, args.analyzer_script, args.config) 
                   for split, task_id in tasks]
    
    # Process tasks in parallel
    with Pool(processes=args.num_processes) as pool:
        pool.map(process_task, process_args)

if __name__ == "__main__":
    main() 