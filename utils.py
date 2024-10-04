import matplotlib.pyplot as plt
import re, os

import torch

def createLossPlot(steps, losses, title="training"):
    plt.plot(steps, losses, linewidth=1)
    plt.xlabel("steps")
    plt.ylabel("losses")
    plt.title(title)
    plt.show()

def save_model_parameters_to_file(model, file_path):
    torch.set_printoptions(profile="full")
    with open(file_path, 'w') as file:
        for name, param in model.named_parameters():
            file.write(f"Name: {name}, Shape: {param.shape}\n")
            file.write(f"{param.data.tolist()}\n\n")
    torch.set_printoptions(profile="default")


def parse_parameters_from_file(file_path):
    parameters = {}
    with open(file_path, 'r') as file:
        content = file.read()
    parameter_blocks = content.strip().split('\n\n')
    for block in parameter_blocks:
        lines = block.strip().split('\n')
        if len(lines) < 2:
            continue
        name_line = lines[0]
        data_lines = lines[1:]
        name_match = re.match(r"Name: (.+), Shape", name_line)
        if not name_match:
            continue
        name = name_match.group(1)
        data_str = ''.join(data_lines)
        data = torch.tensor(eval(data_str))
        parameters[name] = data
    return parameters


def visualize_KANama(file_path, combined_file_path, individual_folder_path):
    parameters = parse_parameters_from_file(file_path)

    n = len(parameters)
    cols = 4  # Number of columns in the plot grid
    rows = (n + cols - 1) // cols  # Calculate number of rows needed

    if not os.path.exists(individual_folder_path):
        os.makedirs(individual_folder_path)

    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 5), dpi=100)

    for ax, (name, param) in zip(axes.flatten(), parameters.items()):
        fig_individual, ax_individual = plt.subplots()
        if param.dim() == 1:
            ax.plot(param.numpy())
            ax.set_title(f'{name} (Bias)')
            ax_individual.plot(param.numpy())
            ax_individual.set_title(f'{name} (Bias)')
        elif param.dim() == 2:
            im = ax.imshow(param.numpy(), aspect='auto', cmap='viridis')
            fig.colorbar(im, ax=ax)
            ax.set_title(f'{name} (Weights)')
            im_individual = ax_individual.imshow(param.numpy(), aspect='auto', cmap='viridis')
            fig_individual.colorbar(im_individual, ax=ax_individual)
            ax_individual.set_title(f'{name} (Weights)')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax_individual.set_xlabel('Dimension 1')
        ax_individual.set_ylabel('Dimension 2')
        
        # Save individual plot
        individual_file_path = os.path.join(individual_folder_path, f"{name.replace('.', '_')}.png")
        fig_individual.savefig(individual_file_path)
        plt.close(fig_individual)

    # Remove empty subplots
    for ax in axes.flatten()[n:]:
        fig.delaxes(ax)

    plt.tight_layout()
    plt.savefig(combined_file_path)
    plt.show()