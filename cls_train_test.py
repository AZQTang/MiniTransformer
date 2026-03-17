import torch
import matplotlib.pyplot as plt
def train_classifier(
    encoder, 
    classifier, 
    train_data_loader,
    device, 
    epochs,
    learning_rate
):
    optimizer = torch.optim.Adam(
        list(encoder.parameters())  + list(classifier.parameters()),
        lr=learning_rate,

    )
    criterion = torch.nn.CrossEntropyLoss()

    epoch_accuracy = []

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in train_data_loader:
            xb, yb = xb.to(device), yb.to(device)

            embeddings, _ = encoder(xb)

            avg_hidden_states = torch.mean(embeddings, dim=1)

            outputs = classifier(avg_hidden_states)

            loss = criterion(outputs, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += yb.size(0)
            correct += (predicted == yb).sum().item()

        epoch_acc = 100 * correct / total
        epoch_accuracy.append(epoch_acc)
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_data_loader):.4f}, Accuracy: {epoch_acc:.2f}%")

    return epoch_accuracy



def test_classifier(
    encoder, 
    classifier, 
    test_data_loader,
    device,
):
    encoder.eval()
    classifier.eval()
    
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for xb, yb in test_data_loader:
            xb, yb = xb.to(device), yb.to(device)

            embeddings, _ = encoder(xb)

            avg_hidden_states = torch.mean(embeddings, dim=1)

            outputs = classifier(avg_hidden_states)

            _, predicted = torch.max(outputs, dim=1)
            total_correct += (predicted == yb).sum().item()
            total_samples += yb.size(0)
    accuracy = 100 * total_correct / total_samples
    encoder.train()
    classifier.train()
    return accuracy


def train_model_with_epochs(
    encoder, 
    classifier, 
    train_loader,
    test_loader, 
    device,
    epochs,
    learning_rate
):
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=learning_rate,
    )
    criterion = torch.nn.CrossEntropyLoss()
    
    train_accuracies = []
    test_accuracies = []

    for epoch in range(epochs):
        # training
        encoder.train()
        classifier.train()
        total_loss = 0.0
        correct_train = 0
        total_train = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            embeddings, _ = encoder(xb)

            avg_hidden_states = torch.mean(embeddings, dim=1)

            outputs = classifier(avg_hidden_states)

            loss = criterion(outputs, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += yb.size(0)
            correct_train += (predicted == yb).sum().item()

        train_acc = 100 * correct_train / total_train
        train_accuracies.append(train_acc)

        # testing
        encoder.eval()
        classifier.eval()
        
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)

                embeddings, _ = encoder(xb)

                avg_hidden_states = torch.mean(embeddings, dim=1)

                outputs = classifier(avg_hidden_states)
                _, predicted = torch.max(outputs, dim=1)
                total_test += yb.size(0)
                correct_test += (predicted == yb).sum().item()

        test_acc = 100 * correct_test / total_test
        test_accuracies.append(test_acc)
        print(f"Epoch {epoch+1}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")

    return train_accuracies, test_accuracies


        
        
    
def plot_training_history(train_accuracies, test_accuracies, epochs):
    """Plot training and test accuracies over epochs."""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs+1), train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
    plt.plot(range(1, epochs+1), test_accuracies, 'r-', label='Test Accuracy', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Test Accuracy over Epochs for CLS')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
   
    for i, (train_acc, test_acc) in enumerate(zip(train_accuracies, test_accuracies), 1):
        if i % 3 == 0 or i == epochs:  
            plt.annotate(f'{train_acc:.1f}', (i, train_acc), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=8, color='blue')
            plt.annotate(f'{test_acc:.1f}', (i, test_acc), textcoords="offset points", 
                        xytext=(0,-15), ha='center', fontsize=8, color='red')
    
    plt.tight_layout()
    plt.savefig('training_history_CLS.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Training history saved as 'training_history_CLS.png'")