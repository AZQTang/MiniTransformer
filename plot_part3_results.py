
import matplotlib.pyplot as plt

def plot_part3_results1(
    train_accuracies_without_alibi,
    test_accuracies_without_alibi,
    train_accuracies_with_alibi,
    test_accuracies_with_alibi,
    epochs,
    save_path="part3_results1.png"
):
    plt.figure(figsize=(12, 8))
    epochs_range = range(1, epochs + 1)
    
    # Plot without alibi
    plt.plot(epochs_range, train_accuracies_without_alibi, 'b-', 
             label='Train Accuracy (traditional position embedding)', linewidth=2, marker='o')
    plt.plot(epochs_range, test_accuracies_without_alibi, 'r-', 
             label='Test Accuracy (traditional position embedding)', linewidth=2, marker='s')
    
    # Plot with alibi
    plt.plot(epochs_range, train_accuracies_with_alibi, 'g-', 
             label='Train Accuracy (AliBi)', linewidth=2, marker='^')
    plt.plot(epochs_range, test_accuracies_with_alibi, 'orange', 
             label='Test Accuracy (AliBi)', linewidth=2, marker='d')
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Traditional Position Embedding vs AliBi')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Set x-axis ticks to show only integers, every 3 epochs
    xticks = list(range(1, epochs + 1, 3))
    if epochs not in xticks:  # Add the last epoch if not already included
        xticks.append(epochs)
    plt.xticks(xticks, [int(x) for x in xticks])
    
    # Annotate all points for each line
    # Without alibi - train (blue) - below line
    for epoch, acc in zip(epochs_range, train_accuracies_without_alibi):
        plt.annotate(f'{acc:.1f}', (epoch, acc), 
                    textcoords="offset points", xytext=(0, -15), 
                    ha='center', fontsize=7, color='blue')
    
    # Without alibi - test (red) - below line
    for epoch, acc in zip(epochs_range, test_accuracies_without_alibi):
        plt.annotate(f'{acc:.1f}', (epoch, acc), 
                    textcoords="offset points", xytext=(0, -15), 
                    ha='center', fontsize=7, color='red')
    
    # With alibi - train (green) - above line
    for epoch, acc in zip(epochs_range, train_accuracies_with_alibi):
        plt.annotate(f'{acc:.1f}', (epoch, acc), 
                    textcoords="offset points", xytext=(0, 10), 
                    ha='center', fontsize=7, color='green')
    
    # With alibi - test (orange) - above line
    for epoch, acc in zip(epochs_range, test_accuracies_with_alibi):
        plt.annotate(f'{acc:.1f}', (epoch, acc), 
                    textcoords="offset points", xytext=(0, 10), 
                    ha='center', fontsize=7, color='orange')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Part 3 results saved as '{save_path}'")

def plot_part3_results2(
    perp_history_without_alibi,
    perp_history_with_alibi,
    save_path="part3_perplexity_results2.png"
):
   
    plt.figure(figsize=(12, 8))
    iterations = perp_history_without_alibi['iterations']
    
    # Plot traditional position embedding (solid lines)
    plt.plot(iterations, perp_history_without_alibi['train_perplexities'], 'b-', 
             label='Train Perplexity (traditional position embedding)', linewidth=2, marker='o')
    
    test_colors = {'hbush': 'red', 'obama': 'green', 'wbush': 'orange'}
    test_markers = {'hbush': 's', 'obama': '^', 'wbush': 'd'}
    for test_name in ['hbush', 'obama', 'wbush']:
        plt.plot(iterations, perp_history_without_alibi['test_perplexities'][test_name], 
                color=test_colors[test_name], linestyle='-',
                label=f'Test {test_name} Perplexity (traditional position embedding)', 
                linewidth=2, marker=test_markers[test_name])
    
    # Plot AliBi (dashed lines with different markers)
    plt.plot(iterations, perp_history_with_alibi['train_perplexities'], 'b--', 
             label='Train Perplexity (AliBi)', linewidth=2, marker='o', markersize=6)
    
    for test_name in ['hbush', 'obama', 'wbush']:
        plt.plot(iterations, perp_history_with_alibi['test_perplexities'][test_name], 
                color=test_colors[test_name], linestyle='--',
                label=f'Test {test_name} Perplexity (AliBi)', 
                linewidth=2, marker=test_markers[test_name], markersize=6)
    
    plt.xlabel('Iteration')
    plt.ylabel('Perplexity')
    plt.title('Traditional Position Embedding vs AliBi')
    # Create legend with proper line style display
    legend = plt.legend(loc='best', fontsize=8, ncol=2, framealpha=0.9)
    # Ensure line styles are visible in legend
    for line in legend.get_lines():
        line.set_linewidth(2.5)
    plt.grid(True, alpha=0.3)
    
    # Annotate only the last point for each line to avoid clutter
    # Traditional position embedding - last point only (below)
    last_iter = iterations[-1]
    last_train_perp = perp_history_without_alibi['train_perplexities'][-1]
    plt.annotate(f'{last_train_perp:.1f}', (last_iter, last_train_perp), 
                textcoords="offset points", xytext=(0, -15), 
                ha='center', fontsize=8, color='blue', fontweight='bold')
    
    for test_name in ['hbush', 'obama', 'wbush']:
        last_test_perp = perp_history_without_alibi['test_perplexities'][test_name][-1]
        plt.annotate(f'{last_test_perp:.1f}', (last_iter, last_test_perp), 
                    textcoords="offset points", xytext=(0, -15), 
                    ha='center', fontsize=8, color=test_colors[test_name], fontweight='bold')
    
    # AliBi - last point only (above)
    last_train_perp_alibi = perp_history_with_alibi['train_perplexities'][-1]
    plt.annotate(f'{last_train_perp_alibi:.1f}', (last_iter, last_train_perp_alibi), 
                textcoords="offset points", xytext=(0, 10), 
                ha='center', fontsize=8, color='blue', fontweight='bold')
    
    for test_name in ['hbush', 'obama', 'wbush']:
        last_test_perp_alibi = perp_history_with_alibi['test_perplexities'][test_name][-1]
        plt.annotate(f'{last_test_perp_alibi:.1f}', (last_iter, last_test_perp_alibi), 
                    textcoords="offset points", xytext=(0, 10), 
                    ha='center', fontsize=8, color=test_colors[test_name], fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Part 3 perplexity results saved as '{save_path}'")

# Encoder_train_accuracies_without_alibi = [42.69, 53.73, 64.29, 78.97, 87.91, 93.21, 97.61, 98.66, 99.19, 98.90, 99.43, 98.80, 98.76, 99.62, 99.90, 99.86, 99.28, 99.47, 99.62]

# Encoder_test_accuracies_without_alibi = [33.33, 47.73, 65.07, 76.40, 83.47, 86.27, 87.20, 88.13, 88.27, 87.20, 87.47, 87.33, 87.07, 88.27, 87.87, 88.27, 88.53, 87.87, 86.67]

# Encoder_train_accuracies_with_alibi = [44.89, 57.27, 69.41, 84.85, 93.74, 97.13, 98.52, 98.85, 99.04, 99.14, 99.24, 99.09, 98.37, 99.00, 99.09, 99.00, 99.38, 99.57, 99.19]

# Encoder_test_accuracies_with_alibi = [50.53, 61.33, 61.60, 84.93, 85.73, 87.87, 88.00, 88.40, 88.40, 87.60, 88.27, 85.87, 87.33, 90.00, 86.53, 87.07, 88.27, 88.67, 87.20]
# plot_part3_results1(
#     Encoder_train_accuracies_without_alibi,
#     Encoder_test_accuracies_without_alibi,
#     Encoder_train_accuracies_with_alibi,
#     Encoder_test_accuracies_with_alibi,
#     19
# )
iterations = [100, 200, 300, 400, 500]

perp_history_without_alibi_dict = {
    'iterations': iterations,
    'train_perplexities': [975.9402, 346.0083, 187.0983, 125.6242, 94.7371],
    'test_perplexities': {
        'hbush': [627.8248, 433.3934, 384.3176, 385.3861, 400.7457],
        'obama': [588.6290, 371.0889, 316.1511, 308.9639, 313.9985],
        'wbush': [658.2049, 441.0070, 398.4675, 395.6989, 428.8640]
    }
}

perp_history_with_alibi_dict = {
    'iterations': iterations,
    'train_perplexities': [910.0126, 296.1914, 166.7751, 117.6178, 82.9882],
    'test_perplexities': {
        'hbush': [563.2233, 391.1520, 374.7686, 371.4634, 390.8487],
        'obama': [526.9617, 349.4467, 322.8490, 322.1317, 332.7671],
        'wbush': [580.7230, 406.5280, 395.5418, 405.5406, 431.3776]
    }
}

plot_part3_results2(
    perp_history_without_alibi_dict,
    perp_history_with_alibi_dict,
    "Figure2.png"
)