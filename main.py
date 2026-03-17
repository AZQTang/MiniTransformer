import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import sys
from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset
from transformer import Encoder, Classifier, Decoder
from cls_train_test import train_model_with_epochs, plot_training_history
from utilities import Utilities
from llm_train_test import train_llm_with_epochs, plot_perplexity_history, evaluate_final_perplexity
from plot_part3_results import plot_part3_results1, plot_part3_results2
seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers


eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500  # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set


## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input
## size of 64, hidden size of 100 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15  # epochs for classifier training


def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data.
    """

    texts = []
    files = os.listdir(directory)
    for filename in files:
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), "r", encoding="utf-8") as file:
            texts.append(file.read())
    return texts


def collate_batch(batch):
    """Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(
        padded_sequences,
        (0, max(0, block_size - padded_sequences.shape[1])),
        "constant",
        0,
    )
    labels = torch.stack(labels)
    return padded_sequences, labels


def compute_classifier_accuracy(classifier, data_loader):
    """Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            outputs = classifier(X)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
        accuracy = 100 * total_correct / total_samples
        classifier.train()
        return accuracy


def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    """Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses = []
    total_loss = 0
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        loss = decoderLMmodel(
            X, Y
        )  # your model should be computing the cross entropy loss
        losses.append(loss.item())
        total_loss += loss.item()
        if len(losses) >= eval_iters:
            break

    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity

def no_argv(tokenizer):
    part1(tokenizer)
    part2(tokenizer)
    
def part1(tokenizer):
    train_CLS_dataset = SpeechesClassificationDataset(
        tokenizer, "speechesdataset/train_CLS.tsv"
    )
    print("Number of samples in the training dataset is", len(train_CLS_dataset))
    train_CLS_loader = DataLoader(
        train_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=True
    )
    # create Encoder
    encoder = Encoder(
        vocab_size=tokenizer.vocab_size,
        n_embd = n_embd,
        n_head = n_head,
        n_layer = n_layer,
        block_size = block_size,
    ).to(device)
    # create Classifier
    classifier = Classifier(
        n_input = n_embd,
        n_hidden = n_hidden,
        n_output = n_output,
    ).to(device)
    test_CLS_dataset = SpeechesClassificationDataset(
        tokenizer, "speechesdataset/test_CLS.tsv"
    )
    test_CLS_loader = DataLoader(
        test_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=False
    )
    encoder_params = sum(p.numel() for p in encoder.parameters())
    classifier_params = sum(p.numel() for p in classifier.parameters())
    total_params = encoder_params + classifier_params
    print(f"Encoder parameters: {encoder_params}, Classifier parameters: {classifier_params}, Total parameters: {total_params}")
    
    
    print("Training....")
    train_accuracies, test_accuracies = train_model_with_epochs(
        encoder, classifier, train_CLS_loader, test_CLS_loader, 
        device, epochs_CLS, learning_rate)

    
    plot_training_history(train_accuracies, test_accuracies, epochs_CLS)
    final_test_accuracy = test_accuracies[-1]
    print(f"\n final test accuracy (Epoch {epochs_CLS}): {final_test_accuracy:.2f}%")
    torch.save(encoder.state_dict(), "encoder.pth")

    # sanity check & plot attention maps
    texts = load_texts("speechesdataset")
    tokenizer = SimpleTokenizer(" ".join(texts))
    
    
    encoder.eval()
    
    utilities = Utilities(tokenizer, encoder)
    with torch.no_grad():
        utilities.sanity_check("And Democrats, we must also admit that fulfilling America's promise will require more than just money.", block_size)


def part2(tokenizer):
    inputfile = "speechesdataset/train_LM.txt"
    with open(inputfile, "r", encoding="utf-8") as f:
        lmtrainText = f.read()
    train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText, block_size)
    print("Number of samples in the training dataset is", len(train_LM_dataset))
    

    train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)
    decoder = Decoder(
        vocab_size=tokenizer.vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        block_size=block_size,
    ).to(device)
    decoder_params = sum(p.numel() for p in decoder.parameters())
    print(f"\nDecoder parameters: {decoder_params:,}")

    print("Training language model")
    # losses = train_language_model(decoder, train_LM_loader, device, max_iters, eval_interval, learning_rate)
    # perplexity = compute_perplexity(decoder, train_LM_loader, eval_iters)
    # print(f"Final train perplexity: {perplexity:.4f}")
    # print("Saving model to decoder.pth")
    # torch.save(decoder.state_dict(), "decoder.pth")

    test_files = {
    'hbush': "speechesdataset/test_LM_hbush.txt",
    'obama': "speechesdataset/test_LM_obama.txt",
    'wbush': "speechesdataset/test_LM_wbush.txt"
}
    test_loaders = {}
    for name, filepath in test_files.items():
        with open(filepath, 'r', encoding='utf-8') as f:
            test_text = f.read()
        test_dataset = LanguageModelingDataset(tokenizer, test_text, block_size)
        test_loaders[name] = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    perp_history = train_llm_with_epochs(decoder, train_LM_loader, test_loaders, device, max_iters, eval_interval, learning_rate)
    plot_perplexity_history(perp_history, "train_history_LM.png")
    final_perplexities = evaluate_final_perplexity(decoder, test_loaders, device)
    print(f"Final perplexities: {final_perplexities}")
    print("Saving model...")
    torch.save(decoder.state_dict(), "decoder.pth")
    decoder.eval()
    utilities = Utilities(tokenizer, decoder)
    with torch.no_grad():
        utilities.sanity_check("Whether or not we stand up for our freedoms, whether or not we respect and enforce the rule of law, that's up to us.", block_size)

def part3(tokenizer):
    # encoder + classifier 
    epochs_CLS = 19
    train_CLS_dataset = SpeechesClassificationDataset(
        tokenizer, "speechesdataset/train_CLS.tsv"
    )
    train_CLS_loader = DataLoader(
        train_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=True
    )
    test_CLS_dataset = SpeechesClassificationDataset(
        tokenizer, "speechesdataset/test_CLS.tsv"
    )
    test_CLS_loader = DataLoader(
        test_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=False
    )
    # without alibi
    # create Encoder
    encoder = Encoder(
        vocab_size=tokenizer.vocab_size,
        n_embd = n_embd,
        n_head = n_head,
        n_layer = n_layer,
        block_size = block_size,
    ).to(device)
    # create Classifier
    classifier = Classifier(
        n_input = n_embd,
        n_hidden = n_hidden,
        n_output = n_output,
    ).to(device)
    
    
    
    
    print("Training encoder+classifier without alibi")
    Encoder_train_accuracies_without_alibi, Encoder_test_accuracies_without_alibi = train_model_with_epochs(
        encoder, classifier, train_CLS_loader, test_CLS_loader, 
        device, epochs_CLS, learning_rate)

    
    
    print(f"\n Encoder+classifier without alibi, final test accuracy (Epoch {epochs_CLS}): {Encoder_test_accuracies_without_alibi[-1]:.2f}%")

    #  with alibi
    # create Encoder
    encoder = Encoder(
        vocab_size=tokenizer.vocab_size,
        n_embd = n_embd,
        n_head = n_head,
        n_layer = n_layer,
        block_size = block_size,
        alibi=True,
    ).to(device)
    # create Classifier
    classifier = Classifier(
        n_input = n_embd,
        n_hidden = n_hidden,
        n_output = n_output,
    ).to(device)
    
    
    
    print("Training encoder+classifier with alibi")
    Encoder_train_accuracies_with_alibi, Encoder_test_accuracies_with_alibi = train_model_with_epochs(
        encoder, classifier, train_CLS_loader, test_CLS_loader, 
        device, epochs_CLS, learning_rate)

    print(f"\n Encoder+classifier with alibi, final test accuracy (Epoch {epochs_CLS}): {Encoder_test_accuracies_with_alibi[-1]:.2f}%")
    
    # Plot comparison
    plot_part3_results1(
        Encoder_train_accuracies_without_alibi,
        Encoder_test_accuracies_without_alibi,
        Encoder_train_accuracies_with_alibi,
        Encoder_test_accuracies_with_alibi,
        epochs_CLS
    )
    
    # decoder 
    inputfile = "speechesdataset/train_LM.txt"
    with open(inputfile, "r", encoding="utf-8") as f:
        lmtrainText = f.read()
    train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText, block_size)
    train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)

    test_files = {
    'hbush': "speechesdataset/test_LM_hbush.txt",
    'obama': "speechesdataset/test_LM_obama.txt",
    'wbush': "speechesdataset/test_LM_wbush.txt"
}
    test_loaders = {}
    for name, filepath in test_files.items():
        with open(filepath, 'r', encoding='utf-8') as f:
            test_text = f.read()
        test_dataset = LanguageModelingDataset(tokenizer, test_text, block_size)
        test_loaders[name] = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print("Training decoder without alibi")
    # without alibi
    decoder = Decoder(
        vocab_size=tokenizer.vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        block_size=block_size,
    ).to(device)
    perp_history_without_alibi = train_llm_with_epochs(decoder, train_LM_loader, test_loaders, device, max_iters, eval_interval, learning_rate)
    # print("Final perplexities without alibi:")
    # for name, perplexity in perp_history_without_alibi.items():
    #     print(f"{name}: {perplexity:.4f}")
    print("Training decoder with alibi")
    # with alibi
    decoder = Decoder(
        vocab_size=tokenizer.vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        block_size=block_size,
        alibi=True,
    ).to(device)
    perp_history_with_alibi = train_llm_with_epochs(decoder, train_LM_loader, test_loaders, device, max_iters, eval_interval, learning_rate)
    
    
def main():

    print("Loading data and creating tokenizer ...")
    texts = load_texts("speechesdataset")
    tokenizer = SimpleTokenizer(" ".join(texts))  # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)
    if len(sys.argv) == 1:
        no_argv(tokenizer)
    else:
        if sys.argv[1] == "part1":
            part1(tokenizer)
        elif sys.argv[1] == "part2":
            part2(tokenizer)
        elif sys.argv[1] == "part3":
            part3(tokenizer)
        

    


    


    

    # for the classification  task, you will train for a fixed number of epochs like this:

    # for epoch in range(epochs_CLS):
    #     for xb, yb in train_CLS_loader:
    #         xb, yb = xb.to(device), yb.to(device)

    #         # CLS training code here

    # for the language modeling task, you will iterate over the training data for a fixed number of iterations like this:
    # for i, (xb, yb) in enumerate(train_LM_loader):
    #     if i >= max_iters:
    #         break
    #     xb, yb = xb.to(device), yb.to(device)
        # LM training code here


if __name__ == "__main__":
    main()
