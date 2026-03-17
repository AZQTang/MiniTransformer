import torch
import math
import matplotlib.pyplot as plt
def train_language_model(model, train_loader, device, max_iters=500, eval_interval=100, learning_rate=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    losses = []

    for i, (xb, yb) in enumerate(train_loader):
        if i >= max_iters:
            break
        xb, yb = xb.to(device), yb.to(device)
        loss = model(xb, yb)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (i + 1) % eval_interval == 0 and i > 0:
            avg_loss = sum(losses[-eval_interval:]) / eval_interval
            perplexity = math.exp(avg_loss) # Perplexity=exp(loss)
            print(f"Step {i + 1}: loss = {avg_loss:.4f}, perplexity = {perplexity:.4f}")


    return losses




def train_llm_with_epochs(
    model, 
    train_loader, 
    test_loaders, # dict of 3 test loaders
    device, 
    max_iters, 
    eval_interval = 100,
    learning_rate = 1e-3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    train_perplexities = []
    test_perplexities = {'hbush': [], 'obama': [], 'wbush': []}
    iterations = []


    model.train()
    total_loss = 0
    step_count = 0

    for i, (xb, yb) in enumerate(train_loader):
        if i>= max_iters:
            break
        xb, yb = xb.to(device), yb.to(device)
        loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        step_count += 1

        if (i + 1) % eval_interval == 0:
            avg_train_loss = total_loss / step_count
            train_perp = math.exp(avg_train_loss)
            train_perplexities.append(train_perp)
            
            model.eval()
            with torch.no_grad():
                for test_name, test_loader in test_loaders.items():
                    test_loss = 0
                    test_steps = 0
                    for test_xb, test_yb in test_loader:
                        test_xb, test_yb = test_xb.to(device), test_yb.to(device)
                        test_loss += model(test_xb, test_yb).item()
                        test_steps += 1
                        if test_steps >= eval_interval:
                            break
                    avg_test_loss = test_loss / test_steps
                    test_perp = math.exp(avg_test_loss)
                    test_perplexities[test_name].append(test_perp)
                
                    
            model.train()
            iterations.append(i + 1)


            print(f"\nStep {i+1}:")
            print(f"  Train Perplexity: {train_perp:.4f}")
            for test_name in test_perplexities:
                print(f"  Test ({test_name}) Perplexity: {test_perplexities[test_name][-1]:.4f}")
            total_loss = 0
            step_count = 0
    
    return {
        'iterations': iterations,
        'train_perplexities': train_perplexities,
        'test_perplexities': test_perplexities
    }


def evaluate_final_perplexity(model, test_loaders, device):
    model.eval()
    final_perplexities = {}
    with torch.no_grad():
        for test_name, test_loader in test_loaders.items():
            test_loss = 0
            count = 0
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = model(xb, yb)
                test_loss += loss
                count += 1
            avg_test_loss = test_loss / count
            perplexity = math.exp(avg_test_loss)
            final_perplexities[test_name] = perplexity
            print(f"Final Perplexity on {test_name}: {perplexity:.4f}")
    model.train()
    return final_perplexities

def plot_perplexity_history(perplexity_history, save_path = "train_history_LM.png"):
    plt.figure(figsize=(12, 8))
    iterations = perplexity_history['iterations']
    plt.plot(iterations, perplexity_history['train_perplexities'], 'b-', 
             label='Training Perplexity', linewidth=2, marker='o')
    colors = {'hbush': 'r-', 'obama': 'g-', 'wbush': 'orange'}
    for test_name, perplexities in perplexity_history['test_perplexities'].items():
        plt.plot(iterations, perplexities, colors[test_name], 
                label=f'Test ({test_name}) Perplexity', linewidth=2, marker='s')
    
    plt.xlabel('Iteration')
    plt.ylabel('Perplexity')
    plt.title('Training and Test Perplexity over Iterations for Language Model')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Annotate all points for training line
    for iter_num, train_perp in zip(iterations, perplexity_history['train_perplexities']):
        plt.annotate(f'{train_perp:.1f}', (iter_num, train_perp), 
                    textcoords="offset points", xytext=(0,-15), 
                    ha='center', fontsize=8, color='blue')
    
    # Annotate all points for test lines
    test_colors = {'hbush': 'red', 'obama': 'green', 'wbush': 'orange'}
    for test_name, perplexities in perplexity_history['test_perplexities'].items():
        # wbush (orange) labels above, others below
        offset_y = 10 if test_name == 'wbush' else -15
        for iter_num, test_perp in zip(iterations, perplexities):
            plt.annotate(f'{test_perp:.1f}', (iter_num, test_perp), 
                        textcoords="offset points", xytext=(0, offset_y), 
                        ha='center', fontsize=8, color=test_colors[test_name])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Training history saved as '{save_path}'")
    


    
