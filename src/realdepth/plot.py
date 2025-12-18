"""
Plotting utilities for RealDepth training visualization.
"""
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")

def save_loss_plots(train_losses, val_losses, steps_per_epoch, output_path):
    """
    Save training and validation loss plots
    """
    if train_losses and val_losses:
        fig, ax = plt.subplots(figsize=(14, 7))

        train_steps, train_vals = zip(*train_losses)

        val_epochs, val_vals = zip(*val_losses)
        # Convert epochs to steps for alignment
        val_steps = [e * steps_per_epoch for e in val_epochs]

        sns.lineplot(x=train_steps, y=train_vals, ax=ax, alpha=0.8, label='Training Loss')
        sns.lineplot(x=val_steps, y=val_vals, ax=ax, alpha=0.8,label='Validation Loss')

        ax.set_xlabel('Training Steps', fontsize=13)
        ax.set_ylabel('Loss Value', fontsize=13)

        # Save
        output_path = Path(output_path)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\nTraining loss plot saved to: {output_path}")


def save_component_plots(train_loss_components, output_path):
    """
    Save components loss plot
    """
    # Create a single plot for all components
    components_to_plot = [k for k, v in train_loss_components.items() if v]
    if not components_to_plot:
        return  # No components to plot

    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot each component
    for component_name in components_to_plot:
        data = train_loss_components[component_name]
        if not data:
            continue

        steps, values = zip(*data)
        label = component_name.replace('_', ' ').title()

        sns.lineplot(x=steps, y=values, ax=ax, label=label, alpha=0.5)

    ax.set_xlabel('Training Steps', fontsize=13)
    ax.set_ylabel('Loss Value', fontsize=13)

    # Save
    output_path = Path(output_path)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Loss components plot saved to: {output_path}")
