import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Conv1d, LayerNorm
import numpy as np

# Positional encoding using sinusoids

def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]

    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


# Start with creating an Attention Head class
class AttentionHead(nn.Module):
    def __init__(self, embedding_dim, head_dim):
        super(AttentionHead, self).__init__()
        self.head_dim = head_dim

        # linear projections to get the weights for the query, key, value for a single head
        # project to the head_dim instead of the embedding_dim
        self.weight_q = nn.Linear(embedding_dim, head_dim)
        self.weight_k = nn.Linear(embedding_dim, head_dim)
        self.weight_v = nn.Linear(embedding_dim, head_dim)

        # then project back to the embedding_dim
        self.weight_head = nn.Linear(head_dim, embedding_dim)
    
    def forward(self, x):
        # x shape: [batch_size, num_patches, embedding_dim]
        
        # project to head dim
        Q = self.weight_q(x)
        K = self.weight_k(x)
        V = self.weight_v(x)
        
        A = torch.einsum('bid,bjd->bij', Q, K)
        #print('Attention weights shape: ', A.shape)
       
        A = torch.softmax(A, dim=-1)

        # apply attention weights to values
        H = torch.einsum('bij,bjd->bid', A, V)

        return H

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # create a list of attention heads
        self.heads = nn.ModuleList(
            [AttentionHead(embedding_dim, self.head_dim) for _ in range(num_heads)]
            )
        #print("Number of heads: ", len(self.heads))

        # create a linear layer to project the concatenated heads back to the embedding dimension
        self.output_linear = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):

        # pass each head through the attention layer to get attention weights
        head_outputs = []
        for head in self.heads:
            head_outputs.append(head(x))

        # concatenate the heads
        # dim=-1 means concatenate along the last dimension
        concat_heads = torch.cat(head_outputs, dim=-1)
        #print('Concatenated heads shape: ', concat_heads.shape)

        # project the concatenated heads back to the embedding dimension
        output = self.output_linear(concat_heads)
        #print('Output shape: ', output.shape)

        return output
    


# now that we have a single Head Attention Layer, and know how to use it in a multi-head context, we can build the encoder block
class EncoderBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, mlp_dimension):
        super(EncoderBlock, self).__init__()

        # normalisation layer before attention
        self.ln1 = nn.LayerNorm(embedding_dim)

        # create a multi-head attention layer
        self.mha = MultiHeadAttention(embedding_dim, num_heads)


        # create a feed forward network
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dimension),  # W₁, b₁
            nn.ReLU(),                               # max(0, x)
            nn.Linear(mlp_dimension, embedding_dim)   # W₂, b₂
        )

        # normalisation layer after attention
        self.ln2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        # first attention block with residual connection
        x = self.ln1(x + self.mha(x))

        # FFN block with residual connection
        x = self.ln2(self.ffn(x) + x)

        return x
    
    

class AudioEncoder(nn.Module):
    def __init__(self, n_mels: int, n_ctx: int,
                 n_state: int, n_head: int, 
                 n_layer: int, mlp_dim: int,
                 num_classes: int, use_classification: bool = True):
        """
        n_mels: number of mel filters
        n_ctx: number of time frames
        n_state: embedding dimension
        n_head: number of attention heads
        n_layer: number of encoder 
        mlp_dim: dimension of the feed-forward network (MLP)
        num_classes: number of output classes for classification
        use_classification: whether to include the classification head or not

        """
        super().__init__()
        self.use_classification = use_classification

        # Convolution layers to process the spectrograms
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)

        # Add positional encoding
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        # Encoder blocks
        self.blocks = nn.ModuleList([EncoderBlock(n_state, n_head, mlp_dim) for _ in range(n_layer)])

        # Layer norm after the encoder blocks
        self.ln_post = LayerNorm(n_state)

        # Classification head
        if self.use_classification:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(n_state),
                nn.Linear(n_state, mlp_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(mlp_dim, num_classes)
            )
            self.softmax = nn.Softmax(dim=-1)
            

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
       
        #print("00:", x.shape)     

        # Pass through convolution layers
        x = F.gelu(self.conv1(x)) 
        #print("after conv1 shape: ", x.shape) # expect [batch size, 384, 345]
        x = F.gelu(self.conv2(x)) 
        #print("after conv2 shape: ", x.shape) # expect [batch size, 384, 173]
        x = x.permute(0, 2, 1)    
        #print("after permute shape: ", x.shape) # expect [batch size, 173, 384]
        
        # Print expected shape for positional embedding
        #print("Expected positional embedding shape: ", self.positional_embedding.shape)
        positional_embedding = self.positional_embedding[:x.shape[1]]

        
        assert x.shape[1:] == positional_embedding.shape, "incorrect audio shape" # [batch size, time steps, n_state]
        x = (x + positional_embedding).to(x.dtype) # [time steps, n_state]
        #print("Mel spectrogram + positional encoding shape: ", x.shape)

        # Pass through encoder blocks
        for block in self.blocks: 
            x = block(x)

        x = self.ln_post(x)

        # If classification is enabled, use the [CLS] token for classification
        if self.use_classification:
            cls_token = x[:, 0]
            logits = self.mlp_head(cls_token)

            if self.training:
                return logits
            else:
                return self.softmax(logits)
            
        # If not using classification, return the full encoded sequence
        return x