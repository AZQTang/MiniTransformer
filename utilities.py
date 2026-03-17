
# from pyexpat import model
import matplotlib.pyplot as plt
import torch
from tokenizer import SimpleTokenizer
#from main import block_size, n_embd, n_head, n_layer
from transformer import Encoder
# from main import load_texts



class Utilities:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def sanity_check(self, sentence, block_size):
        # Encode the sentence using the tokenizer
        wordids = self.tokenizer.encode(sentence)

        # Prepare the padded input for the model
        padded_sentence = wordids[:block_size] + [0] * (block_size - len(wordids))
        device = next(self.model.parameters()).device
        input_tensor = torch.tensor(padded_sentence, dtype=torch.long).unsqueeze(0).to(device)
        real_token_count = min(len(wordids), block_size)
        # Display input tensor shape
        print("Input tensor shape:", input_tensor.shape) # (1, block_size)

        # Process the input tensor through the encoder model
        _,  attn_maps = self.model(input_tensor) # Ignore the output of the model, and only get the attention maps; make sure your encoder returns the attention maps

        # Display the number of attention maps
        print("Number of attention maps:", len(attn_maps))

        # Visualize and save the attention maps
        for j, attn_map in enumerate(attn_maps):
            # att_map = attn_map.squeeze(0).detach().cpu().numpy()  # Remove batch dimension and convert to NumPy array
            n_heads = attn_map.size(1)
            for head in range(n_heads):
                single_head_attn = attn_map[0, head, :real_token_count, :real_token_count].detach().cpu().numpy()
                # single_head_attn = attn_map[0, head, :, :].detach().cpu().numpy()
                total_prob_over_rows = torch.sum(attn_map[0, head, :, :], dim=1)
                if torch.any(total_prob_over_rows < 0.99) or torch.any(total_prob_over_rows > 1.01):
                    print("Failed normalization test: probabilities do not sum to 1.0 over rows")
                    print("Total probability over rows:", total_prob_over_rows.numpy())

                # Create a heatmap of the attention map
                fig, ax = plt.subplots()
                cax = ax.imshow(single_head_attn, cmap='hot', interpolation='nearest')
                ax.xaxis.tick_top()  
                #tokens = sentence.split()[:real_token_count]
                tokens = []
                for i in range(real_token_count):
                    token = self.tokenizer.itos.get(wordids[i], '<unk>')
                    tokens.append(token)
                print(tokens)
                #step = max(1, block_size // 16)
                ax.set_xticks(range(real_token_count))
                ax.set_yticks(range(real_token_count))
                ax.set_xticklabels(tokens, rotation=90, fontsize=8)
                ax.set_yticklabels(tokens, fontsize=8)
                
                fig.colorbar(cax, ax=ax)
                plt.title(f"Layer {j+1}, Head {head+1} Attention")
                
                # Save the plot
                plt.savefig(f"attention_map_L{j+1}_H{head+1}.png", dpi=300, bbox_inches='tight')
                plt.show()
                plt.close()
                # att_map = attn_map[:, head, :, :].squeeze(0).detach().cpu().numpy()
            # Check if the attention probabilities sum to 1 over rows
            #total_prob_over_rows = torch.sum(attn_map[0], dim=1)
            # if torch.any(total_prob_over_rows < 0.99) or torch.any(total_prob_over_rows > 1.01):
            #     print("Failed normalization test: probabilities do not sum to 1.0 over rows")
            #     print("Total probability over rows:", total_prob_over_rows.numpy())

            # Create a heatmap of the attention map
            # fig, ax = plt.subplots()
            # cax = ax.imshow(att_map, cmap='hot', interpolation='nearest')
            # ax.xaxis.tick_top()  
            # fig.colorbar(cax, ax=ax)  
            # plt.title(f"Attention Map {j + 1}")
            
            # # Save the plot
            # plt.savefig(f"attention_map_{j + 1}.png")
            
            # # Show the plot
            # plt.show()
            
# def main():
#     texts = load_texts("speechesdataset")
#     tokenizer = SimpleTokenizer(" ".join(texts))
    
#     encoder = Encoder(
#         vocab_size=tokenizer.vocab_size,
#         n_embd=n_embd,
#         n_head=n_head,
#         n_layer=n_layer,
#         block_size=block_size,
#     )
#     encoder.load_state_dict(torch.load("encoder.pth"))
#     encoder.eval()
    
#     utilities = Utilities(tokenizer, encoder)
#     with torch.no_grad():
#         utilities.sanity_check("And Democrats, we must also admit that fulfilling America's promise will require more than just money.", block_size)

    

if __name__ == "__main__":
    print("_________")
    #main()
