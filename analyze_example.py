import os
import pickle
from tqdm import tqdm
import argparse
import yaml
from typing import Dict, Any

import numpy as np
import torch
import matplotlib.pyplot as plt



"""
This file allows you to train one model on one task, and see plots of what
the process and end result looks like. You can input the training split and
the task code, and it will:
- Train a model for 1500 steps,
- Plot sampled solutions from the model at every 50 steps,
- Plot the KL and reconstruction error over time,
- Plot the contribution of each tensor shape to the KL over time,
- Show top principal components of each tensor that still contributes to
  the KL at the end of training.
"""



def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_default_config() -> Dict[str, Any]:
    """Get default configuration."""
    return {
        'cli_args': {
            'split': None,
            'tasks': None,
            'task': None,
            'output_dir': 'outputs',
            'device': 'cuda:0'
        },
        'model': {
            'n_layers': 4,
            'share_up_dim': 16,
            'share_down_dim': 8,
            'decoding_dim': 4,
            'softmax_dim': 2,
            'cummax_dim': 4,
            'shift_dim': 4,
            'nonlinear_dim': 16
        },
        'optimizer': {
            'lr': 0.01,
            'betas': [0.5, 0.9]
        },
        'training': {
            'n_iterations': 1500,
            'plot_interval': 50
        }
    }

if __name__ == "__main__":
    # Parse args and load config first
    parser = argparse.ArgumentParser(description='Analyze ARC task training and visualization')
    parser.add_argument('--split', type=str, choices=['training', 'evaluation', 'test'],
                      help='which split to find the task in (training, evaluation, test)')
    parser.add_argument('--task', type=str,
                      help='task ID to analyze (eg. 272f95fa)')
    parser.add_argument('--output-dir', type=str, default='outputs',
                      help='directory to store output files (default: outputs)')
    parser.add_argument('--config', type=str,
                      help='path to YAML config file')
    parser.add_argument('--device', type=str, default='cuda:0',
                      help='device to run on (default: cuda:0)')
    args = parser.parse_args()

    # Fix the config loading logic
    if args.config:
        # Load the config file first
        config = load_config(args.config)
    else:
        # If no config file, use defaults
        config = get_default_config()

    # Then apply any command line overrides
    if args.split:
        config['cli_args']['split'] = args.split
    if args.task:
        config['cli_args']['task'] = args.task
        config['cli_args']['tasks'] = [args.task]
    if args.output_dir != 'outputs':  # Only override if explicitly set
        config['cli_args']['output_dir'] = args.output_dir
    if args.device != 'cuda:0':  # Only override if explicitly set
        config['cli_args']['device'] = args.device

    # Set device BEFORE importing other modules
    torch.set_default_device(config['cli_args']['device'])

    import train
    import preprocessing
    import arc_compressor
    import initializers
    import multitensor_systems
    import layers
    import solution_selection
    import visualization

    # For some reason trying to set the seed doesn't actually fix results.
    # Just run things over and over again until you see desired interesting behaviors.
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float32)

    # Use config values or prompt user
    split = config['cli_args']['split']
    if not split:
        split = input('Enter which split you want to find the task in (training, evaluation, test): ')
    
    task_name = config['cli_args']['task']
    if not task_name:
        if config['cli_args']['tasks']:
            task_name = config['cli_args']['tasks'][0]
        else:
            task_name = input('Enter which task you want to analyze (eg. 272f95fa): ')

    # Use the specified output directory
    folder = f"{config['cli_args']['output_dir'].rstrip('/')}/{task_name}/"
    print('Performing a training run on task', task_name,
          'and placing the results in', folder)
    os.makedirs(folder, exist_ok=True)

    # Preprocess the task, set up the training
    task = preprocessing.preprocess_tasks(split, [task_name])[0]
    
    # Create model with config
    model = arc_compressor.ARCCompressor(task, **config['model'])
    
    # Create optimizer with config
    optimizer = torch.optim.Adam(
        model.weights_list, 
        lr=config['optimizer']['lr'], 
        betas=tuple(config['optimizer']['betas'])
    )
    
    train_history_logger = solution_selection.Logger(task)
    visualization.plot_problem(train_history_logger)

    # Perform training using config
    n_iterations = config['training']['n_iterations']
    plot_interval = config['training']['plot_interval']
    
    for train_step in tqdm(range(n_iterations)):
        train.take_step(task, model, optimizer, train_step, train_history_logger)
        
        # Plot solutions at specified interval
        if (train_step+1) % plot_interval == 0:
            visualization.plot_solution(train_history_logger,
                fname=folder + task_name + '_at_' + str(train_step+1) + ' steps.png')
            visualization.plot_solution(train_history_logger,
                fname=folder + task_name + '_at_' + str(train_step+1) + ' steps.pdf')

    # Save the metrics, model weights, and learned representations.
    np.savez(folder + task_name + '_KL_curves.npz',
             KL_curves={key:np.array(val) for key, val in train_history_logger.KL_curves.items()},
             reconstruction_error_curve=np.array(train_history_logger.reconstruction_error_curve),
             multiposteriors=model.multiposteriors,
             target_capacities=model.target_capacities,
             decode_weights=model.decode_weights)

    # Load the metrics, model weights, and learned representations.
    stored_data = np.load(folder + task_name + '_KL_curves.npz', allow_pickle=True)
    KL_curves = stored_data['KL_curves'][()]
    reconstruction_error_curve = stored_data['reconstruction_error_curve']
    multiposteriors = stored_data['multiposteriors'][()]
    target_capacities = stored_data['target_capacities'][()]
    decode_weights = stored_data['decode_weights'][()]
    
    # Plot the KL curves over time.
    # For specific tasks that we found interesting, we wrote some
    # code to color interesting KL components differently.
    special_curve_colors = {
            '272f95fa': {
                'dims': [(1,0,0,1,0), (1,0,0,0,1), (0,1,1,0,0), (0,1,0,0,0)],
                'colors': [(1, 0, 0), (0, 1, 0), (0, 0.5, 1), (0.5, 0, 1)]
            },
            '6cdd2623': {
                'dims': [(1,0,0,1,0), (1,0,0,0,1), (1,1,0,0,0), (1,0,0,1,1), (0,0,1,0,0)],
                'colors': [(1, 0.6, 0), (0, 1, 0), (0, 0.5, 1), (0.5, 0, 1), (1, 0, 0.5)]
            },
            '41e4d17e': {
                'dims': [(1,0,0,1,1), (0,1,0,0,0)],
                'colors': [(1, 0, 0), (0, 0, 1)]
            },
            '6d75e8bb': {
                'dims': [(1,0,0,1,0), (1,0,0,0,1), (1,0,0,1,1), (0,1,0,0,0)],
                'colors': [(1, 0, 0), (0, 1, 0), (0, 0.5, 1), (0.5, 0, 1)]
            }
    }
    fig, ax = plt.subplots()
    for component_name, curve in KL_curves.items():
        line_color = (0.5, 0.5, 0.5)
        label = None
        if task_name in special_curve_colors:
            dims_list = special_curve_colors[task_name]['dims']
            colors_list = special_curve_colors[task_name]['colors']
            for dims, color in zip(dims_list, colors_list):
                if tuple(eval(component_name)) == dims:
                    line_color = color
                    axis_names = ['example', 'color', 'direction', 'height', 'width']
                    axis_names = [axis_name
                        for axis_name, axis_exists in zip(axis_names, dims) if axis_exists]
                    label = '(' + ', '.join(axis_names) + ', channel)'
        ax.plot(np.arange(curve.shape[0]), curve, color=line_color, label=label)
    if task_name == '6cdd2623':
        ax.set_ylim((0.3, 4e4))

    ax.legend()
    plt.yscale('log')
    plt.xlabel('step')
    plt.ylabel('KL contribution')
    ax.grid(which='both', linestyle='-', linewidth='0.5', color='gray')
    plt.savefig(folder + task_name + '_KL_components.png', bbox_inches='tight')
    plt.close()

    # Plot the KL vs reconstruction error
    fig, ax = plt.subplots()
    total_KL = 0
    for component_name, curve in KL_curves.items():
        total_KL = total_KL + curve
    fig, ax = plt.subplots()
    ax.plot(np.arange(total_KL.shape[0]), total_KL, label='KL from z', color='k')
    ax.plot(np.arange(reconstruction_error_curve.shape[0]),
            reconstruction_error_curve, label='reconstruction error', color='r')
    ax.legend()
    plt.yscale('log')
    plt.xlabel('step')
    plt.ylabel('total KL or reconstruction error')
    ax.grid(which='both', linestyle='-', linewidth='0.5', color='gray')
    plt.savefig(folder + task_name + '_KL_vs_reconstruction.png', bbox_inches='tight')
    plt.close()

    # Get the learned representation tensors
    samples = []
    for i in range(100):
        sample, KL_amounts, KL_names = layers.decode_latents(target_capacities,
                                           decode_weights, multiposteriors)
        samples.append(sample)

    def average_samples(dims, *items):
        mean = torch.mean(torch.stack(items, dim=0), dim=0).detach().cpu().numpy()
        all_but_last_dim = tuple(range(len(mean.shape) - 1))
        mean = mean - np.mean(mean, axis=all_but_last_dim)
        return mean
    means = multitensor_systems.multify(average_samples)(*samples)

    # Figure out which tensors contain significant information
    dims_to_plot = []
    for KL_amount, KL_name in zip(KL_amounts, KL_names):
        dims = tuple(eval(KL_name))
        if torch.sum(KL_amount).detach().cpu().numpy() > 1:
            dims_to_plot.append(dims)

    # Show the top principal components of the significant tensors.
    color_names = ['black', 'blue', 'red', 'green', 'yellow', 'gray', 'magenta', 'orange', 'light blue', 'brown']
    restricted_color_names = [color_names[i] for i in task.colors]
    restricted_color_codes = [tuple((visualization.color_list[i]/255).tolist())
                              for i in task.colors]
    for dims in dims_to_plot:
        tensor = means[dims]

        orig_shape = tensor.shape
        if len(orig_shape) == 2:
            tensor = tensor[None,:,:]
        orig_shape = tensor.shape
        if len(orig_shape) == 3:
            tensor = np.reshape(tensor, (-1, orig_shape[-1]))
            U, S, Vh = np.linalg.svd(tensor)  # Get top 3 principal components
            for component_num in range(3):
                component = np.reshape(U[:,component_num], orig_shape[:-1])
                component = component / np.max(np.abs(component))
                strength = S[component_num] / tensor.shape[0]  # Calculate component strength

                # Show the component
                fig, ax = plt.subplots()
                ax.imshow(component, cmap='gray', vmin=-1, vmax=1)
                
                # Pick the axis labels
                axis_names = ['example', 'color', 'direction', 'height', 'width']
                tensor_name = '_'.join([axis_name
                    for axis_name, axis_exists in zip(axis_names, dims) if axis_exists])
                if sum(dims) == 2:
                    x_dim = [axis_names[i] for i, dim in enumerate(dims) if dim][0]
                    y_dim = [axis_names[i] for i, dim in enumerate(dims) if dim][1]
                else:
                    x_dim = None
                    y_dim = [axis_names[i] for i, dim in enumerate(dims) if dim][0]
                plt.ylabel(x_dim)
                plt.xlabel(y_dim)

                if x_dim is None:
                    ax.set_yticks([])
                    ax.set_xticks([], minor=True)
                if y_dim is None:
                    ax.set_xticks([])
                    ax.set_xticks([], minor=True)

                # Set the tick labels
                # Tick labels for example axis
                if x_dim == 'example':
                    ax.set_yticks(np.arange(task.n_examples))
                if y_dim == 'example':
                    ax.set_xticks(np.arange(task.n_examples))

                # Tick labels for color axis
                if x_dim == 'color':
                    ax.set_yticks(np.arange(len(restricted_color_names[1:])))
                    ax.set_yticklabels(restricted_color_names[1:])
                    for ticklabel, tickcolor in zip(ax.get_yticklabels(), restricted_color_codes[1:]):
                        ticklabel.set_color(tickcolor)
                        ticklabel.set_fontweight("bold")
                if y_dim == 'color':
                    ax.set_xticks(np.arange(len(restricted_color_names[1:])))
                    ax.set_xticklabels(restricted_color_names[1:])
                    for ticklabel, tickcolor in zip(ax.get_xticklabels(), restricted_color_codes[1:]):
                        ticklabel.set_color(tickcolor)
                        ticklabel.set_fontweight("bold")

                # Tick labels for direction axis
                direction_names = ["↓", "↘", "→", "↗", "↑", "↖", "←", "↙"]
                if x_dim == 'direction':
                    ax.set_yticks(np.arange(8))
                    ax.set_yticklabels(direction_names)
                    ax.tick_params(axis='y', which='major', labelsize=22)
                if y_dim == 'direction':
                    ax.set_xticks(np.arange(8))
                    ax.set_xticklabels(direction_names)
                    ax.tick_params(axis='x', which='major', labelsize=22)

                # Standard tick labels for height and width axes

                ax.set_title('component' + str(component_num) + ', strength = ' + str(float(strength)))
                plt.savefig(folder + task_name + '_' + tensor_name + '_component_' + str(component_num) + '.png', bbox_inches='tight')
                plt.close()

        # Plot an ({example, color, direction}, x, y) tensor with subplots
        elif len(orig_shape) == 4 and dims[3] == 1 and dims[4] == 1:
            tensor = np.reshape(tensor, (-1, orig_shape[-1]))
            U, S, Vh = np.linalg.svd(tensor)  # Get the top 3 principal components
            for component_num in range(3):
                component = np.reshape(U[:,component_num], orig_shape[:-1])
                component = component / np.max(np.abs(component))
                strength = S[component_num] / tensor.shape[0]
                n_plots = orig_shape[0]

                # Make the subplots
                fig, axs = plt.subplots(1, n_plots)
                for plot_idx in range(n_plots):
                    ax = axs[plot_idx]
                    ax.imshow(component[plot_idx,:,:], cmap='gray', vmin=-1, vmax=1)

                    # Get the axis labels
                    axis_names = ['example', 'color', 'direction', 'height', 'width']
                    tensor_name = '_'.join([axis_name
                        for axis_name, axis_exists in zip(axis_names, dims) if axis_exists])
                    ax_dim = [axis_names[i] for i, dim in enumerate(dims) if dim][0]
                    x_dim = [axis_names[i] for i, dim in enumerate(dims) if dim][1]
                    y_dim = [axis_names[i] for i, dim in enumerate(dims) if dim][2]
                    ax.set_ylabel(x_dim)
                    ax.set_xlabel(y_dim)

                    # Standard tick labels for height and width axes

                    # Label the subplots
                    if ax_dim == 'example':
                        ax.set_title('example ' + str(plot_idx))
                    elif ax_dim == 'color':
                        ax.set_title(restricted_color_names[plot_idx],
                                     color=restricted_color_codes[plot_idx],
                                     fontweight="bold")
                    elif ax_dim == 'direction':
                        direction_names = ["↓", "↘", "→", "↗", "↑", "↖", "←", "↙"]
                        ax.set_title(direction_names[plot_idx], fontsize=22)

                plt.subplots_adjust(wspace=1)
                fig.suptitle('component ' + str(component_num) + ', strength = ' + str(float(strength)))
                plt.subplots_adjust(top=1.4)
                plt.savefig(folder + task_name + '_' + tensor_name + '_component_' + str(component_num) + '.png', bbox_inches='tight')
                plt.close()


print('done')
