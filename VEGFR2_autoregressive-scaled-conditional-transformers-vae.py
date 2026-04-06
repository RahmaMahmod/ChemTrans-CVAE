#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
np.random.seed( 42 )
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors, Lipinski ,QED,rdMolDescriptors,RDConfig
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
# now you can import sascore!
import sascorer
import selfies as sf
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms
import lightning.pytorch as pl
from torch.optim.lr_scheduler import StepLR
import torch
import torch.nn as nn
# import pytorch_lightning as pl
from transformers import AutoModelForCausalLM, AutoTokenizer
import math


# In[6]:


SELFIES = pd.read_csv(r"/home/rahma/Data_and_code/VEGFR2/VEGFR2_pref_name_SELFIES.csv")
SELFIES = SELFIES["SELFIES"]
SELFIES


# In[7]:


canonical_smiles = pd.read_csv(r"/home/rahma/Data_and_code/VEGFR2/VEGFR2_pref_name_Smiles.csv")
canonical_smiles = canonical_smiles["canonical_smiles"]
canonical_smiles


# # Test selfies

# In[9]:


type(SELFIES.tolist())


# # Tokenizaion

# In[10]:


# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("cloudrambler/selfiesbert")


# In[11]:


input_ids = tokenizer(SELFIES.tolist(),return_tensors="pt",padding=True)
input_ids


# In[12]:


# --- constants coming from your tokenizer -----------------
tokenizer.convert_tokens_to_ids('[CLS]'),tokenizer.convert_tokens_to_ids('[SEP]'),tokenizer.convert_tokens_to_ids('[PAD]') 


# In[13]:


input_ids = input_ids['input_ids']
# non_padding_lengths = (input_ids != tokenizer.pad_token_id).sum(dim=1)
# non_padding_lengths


# In[14]:


# np.save('input_ids.npy',input_ids['input_ids'])


# In[15]:


# input_ids = np.load('input_ids.npy',mmap_mode='r')


# In[16]:


input_ids.shape


# In[17]:


len(tokenizer.vocab)


# In[22]:



selected_desc = pd.read_csv(r"/home/rahma/Data_and_code/VEGFR2/VEGFR2_pref_name_selected_descriptors.csv")
selected_desc


# In[26]:


selected_desc_columns = selected_desc.columns


# In[27]:


# selected_desc.describe()


# In[28]:


# selected_desc.to_csv(r"selected_desc.csv", index=False)


# In[29]:


selected_desc=np.array(selected_desc)
type(selected_desc)


# In[30]:


featuer_len=selected_desc.shape[1]
featuer_len


# In[31]:


desirable_descriptors= pd.read_csv(r"/home/rahma/Data_and_code/VEGFR2/VEGFR2_pref_name_desirable_descriptors.csv")
desirable_descriptors


# In[32]:


num_desirable_descriptors = desirable_descriptors.shape[1]


# # try scale desirable_descriptors and compare between the result

# In[33]:


# from sklearn import preprocessing
# des_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
# des_scaler.fit(desirable_descriptors)
# scaled_desirable_descriptors = des_scaler.transform(desirable_descriptors)
# scaled_desirable_descriptors


# In[34]:


import joblib
# Save the fitted scaler
# joblib.dump(des_scaler, 'minmax_scaler_desirable_decoder_descriptors.pkl')


# # Load the scaler
loaded_scaler = joblib.load('minmax_scaler_desirable_decoder_descriptors.pkl')

# Transform new data
scaled_desirable_descriptors = loaded_scaler.transform(desirable_descriptors)


# # Implement a custom Dataset:
# * **inherit Dataset**
# * **implement __init__ , __getitem__ , and __len__**
# 

# In[35]:


class EncodingSELFIES_Dataset(Dataset):

    def __init__(self,tokens,canonical_smiles,descriptor,desirable_descriptors):

        self.tokens = tokens
        self.descriptor = torch.Tensor(descriptor)
#         print(descriptor)

        self.canonical_smiles = np.array(canonical_smiles)
        self.desirable_descriptors = np.array(desirable_descriptors).astype(float)
        

    def __getitem__(self,index):
        return self.tokens[index],self.canonical_smiles[index], self.descriptor[index],self.desirable_descriptors[index]


    # we can call len(dataset) to return the size
    def __len__(self):
        return len(self.tokens)


# In[36]:


input_ids.shape ,canonical_smiles.shape,selected_desc.shape,desirable_descriptors.shape


# In[37]:


# create dataset
SELFIES_Dataset = EncodingSELFIES_Dataset(tokens=input_ids,
                                          canonical_smiles=canonical_smiles
                                          ,descriptor=selected_desc
                                          ,desirable_descriptors = scaled_desirable_descriptors
                                          )


# In[38]:


train_loader = DataLoader(dataset=SELFIES_Dataset,
                          batch_size= 64,
                          shuffle=True)


# In[39]:


# convert to an iterator and look at one random sample
dataiter = iter(train_loader)
dataiter


# In[40]:


data = next(dataiter)
tokens,smiles,descriptor,desirable_desc = data
tokens,smiles,descriptor,desirable_desc


# In[41]:


# tokens[0],smiles[0],model_out_to_smiles(tgt_one_hot)[0]


# In[42]:


# calculate_similarity(model_out_to_smiles(tgt_one_hot)[20], smiles[20], 'Morgan')


# # important functions

# In[43]:


def calculate_similarity(smi1, smi2, fingerprint_type):
    mol1 = Chem.MolFromSmiles(smi1)
    mol2 = Chem.MolFromSmiles(smi2)
    if fingerprint_type == 'Morgan':
        # Generate Morgan fingerprints
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
        try:
          fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
        except Exception as e:
          return None
    elif fingerprint_type == 'MACCSkeys':
        # Generate MACCSkeys fingerprints
        fp1 = MACCSkeys.GenMACCSKeys(mol1)
        try:
          fp2 = MACCSkeys.GenMACCSKeys(mol2)
        except Exception as e:
          return None
    elif fingerprint_type == 'PubChem':
        # Generate PubChem fingerprints
        fp1 = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol1)
        try:
          fp2 = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol2)
        except:
          return None

    else:
        raise ValueError("Invalid fingerprint type specified.")

    # Calculate similarity
    similarity = DataStructs.TanimotoSimilarity(fp1, fp2)


    return similarity


# In[44]:


import re

def remove_spaces_between_brackets(text):
    pattern = r"\]\s+|\s+\["
    return re.sub(pattern, lambda m: m.group().replace(' ', ''), text)


# In[45]:


def model_out_to_smiles(yhat):
  _, topi = yhat.topk(1)
  decoded_ids = topi.squeeze()

  reconstructed_molecules = []
  for i in decoded_ids:
    recons_selfes = tokenizer.decode(i,skip_special_tokens=True)

    recons_selfes = remove_spaces_between_brackets(recons_selfes)

    reconstructed_molecules.append(sf.decoder(recons_selfes))
  return reconstructed_molecules


# In[46]:


def rmsd(coordinates1, coordinates2):

    _, topi = coordinates1.topk(1)
    decoded_ids = topi.squeeze()

    # Convert coordinates to NumPy arrays
    coordinates1 = np.array(decoded_ids.cpu())
    coordinates2 = np.array(coordinates2.cpu())

    # Calculate the difference between the coordinates
    diff = coordinates1 - coordinates2

    # Calculate the sum of squared differences
    squared_diff = np.sum(diff**2, axis=1)

    # Calculate the mean of squared differences
    mean_squared_diff = np.mean(squared_diff)

    # Calculate the RMSD
    rmsd = np.sqrt(mean_squared_diff)

    return rmsd


# ---
# ---
# 
# # Variational AutoEncoders

# In[47]:


class EncoderVar(nn.Module):
    def __init__(self, z_size, base_model):
        super().__init__()

        self.z_size = z_size
        # self.input_shape = input_shape
        self.base_model = base_model
#         output_size = base_model.hidden_size
        # output_size = self.get_output_size()
        output_size =768

        self.lin_mu = nn.Linear(output_size, z_size)
        self.lin_var = nn.Linear(output_size, z_size)


    # def get_output_size(self):
    #     size = self.base_model(torch.zeros(1, *self.input_shape, device= self.device)).size(1)
    #     return size

    def kl_loss(self):
        kl_loss = -0.5*(1 + self.log_var - self.mu**2 - torch.exp(self.log_var))
        return kl_loss

#     def kl_loss(self):
#         prior = torch.distributions.Normal(0, 1)
#         # Define the posterior distribution based on the encoder output
#         posterior = torch.distributions.Normal( self.mu, torch.exp(0.5 * self.log_var))
#         # Compute the KL divergence between the prior and posterior distributions
#         kl_div = torch.distributions.kl_divergence(posterior, prior)#.sum(dim=1)

#         return kl_div

    def forward(self, x,descriptor):
#         import pdb
#         pdb.set_trace()
        # the base model, same as the traditional AE
#         base_out = self.base_model(x,data_lengths.to('cpu'))
        base_out = self.base_model(x,descriptor)#[0]

        # now the encoder produces means (mu) using the lin_mu output layer
        # and log variances (log_var) using the lin_var output layer
        # we compute the standard deviation (std) from the log variance
        self.mu = self.lin_mu(base_out)
        self.log_var = self.lin_var(base_out)
        std = torch.exp(self.log_var/2)

        # that's the internal random input (epsilon)
        eps = torch.randn_like(self.mu)
        # and that's the z vector
        self.z = self.mu + eps * std

        return self.z


# In[48]:


from transformers import AutoModelForMaskedLM
model = AutoModelForMaskedLM.from_pretrained("cloudrambler/selfiesbert")


# In[49]:


model.config.vocab_size


# In[50]:


model.config


# In[51]:


model.base_model.embeddings


# In[52]:


class BertEncoder(nn.Module):
    def __init__(self,model):
        super().__init__()


        self.embed_tokens= model.base_model.embeddings

#         self.lin_z = nn.Linear(1024+120, 1024)

        self.cnn_Encoder = nn.Conv1d(768 + featuer_len, 768, kernel_size=3, padding=1)

        self.Bert_encoder =model.base_model.encoder


    def forward(self,x,descriptor):
#         import pdb
#         pdb.set_trace()
        # Create attention mask tensor of ones
#         attention_mask = torch.ones(x.size(0),1,x.size(1),x.size(1)).bool().to("cuda")

        attention_mask = x.ne(tokenizer.pad_token_id).unsqueeze(1).unsqueeze(2).to(torch.bool).to(x.device)
        # Create empty layer_head_mask tensor
        layer_head_mask =None

        x= self.embed_tokens(x)

        descriptor = descriptor.unsqueeze(1)
        descriptor = descriptor.repeat(1, x.size(1), 1)

        x = torch.cat((x, descriptor), dim=-1)
#         x = self.lin_z(x)

        # Reshape x for CNN layer
        x = x.permute(0, 2, 1)  # Reshape to (batch_size, features, sequence_length)

        x = self.cnn_Encoder(x)

        # Reshape x back to original shape
        x = x.permute(0, 2, 1)  # Reshape back to (batch_size, sequence_length, features)

        x= self.Bert_encoder(x, attention_mask, layer_head_mask)

        return x[0]


# In[53]:


enc = BertEncoder(model)


# In[106]:


class AutoregressiveDecoder(nn.Module):
    def __init__(self, vocab_size, z_size, num_descriptors, max_seq_len=256, d_model=768, nhead=12, num_layers=6, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.z_size = z_size
        self.num_descriptors = num_descriptors
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

        self.embed_tokens= model.base_model.embeddings
        
        # Z condition projection
        self.z_proj = nn.Linear(z_size + num_descriptors, d_model)

        # self.cnn_Decoder = nn.Conv1d(z_size
        #                      + num_descriptors
        #                      , d_model, kernel_size=3, padding=1)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            #layer_norm_eps = 1e-12,
            activation='gelu',
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def create_causal_mask(self, seq_len, device):
        """Create a causal (lower triangular) mask for autoregressive generation"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.bool()
    
    def forward(self, z, descriptor, target_tokens=None, max_length=None):
        """
        Args:
            z: latent representation [batch_size, seq_len, z_size]
            descriptor: molecular descriptors [batch_size, num_descriptors]
            target_tokens: target sequence for training [batch_size, seq_len] (optional)
            max_length: maximum generation length for inference (optional)
        """

        batch_size = z.size(0)
        device = z.device
        
        # Prepare condition (z + descriptor)
        descriptor = descriptor.unsqueeze(1).repeat(1, z.size(1), 1)
        condition = torch.cat([z, descriptor], dim=-1).to(self.z_proj.weight.dtype) 
        
        memory = self.z_proj(condition)  # [batch_size, seq_len, d_model]

        # # Apply CNN layer
        # memory = condition.permute(0, 2, 1)  # Reshape to (batch_size, d_model, sequence_length)
        # memory = self.cnn_Decoder(memory.float())

        # memory = memory.permute(0, 2, 1)  # Reshape back to (batch_size, sequence_length, d_model)


        
        if target_tokens is not None:
            # Training mode - teacher forcing
            seq_len = target_tokens.size(1)
            
            token_emb = self.embed_tokens(target_tokens)
            
            # Create causal mask
            tgt_mask = self.create_causal_mask(seq_len, device)
            
            # Transformer decoder
            output = self.transformer_decoder(
                tgt=token_emb,
                memory=memory,
                tgt_mask=tgt_mask
            )
            
            # Project to vocabulary
            logits = self.output_proj(output)
            return logits
        
        else:
            # Inference mode - autoregressive generation
            if max_length is None:
                max_length = self.max_seq_len
            
            # Start with CLS token (assuming token id 1 is CLS)
            generated = torch.ones(batch_size, tokenizer.cls_token_id, dtype=torch.long, device=device)
            
            for _ in range(max_length - 1):
                token_emb = self.embed_tokens(generated)

                # Create causal mask
                seq_len = generated.size(1)
                tgt_mask = self.create_causal_mask(seq_len, device)
                
                # Transformer decoder
                output = self.transformer_decoder(
                    tgt=token_emb,
                    memory=memory,
                    tgt_mask=tgt_mask
                )
                
                # Get logits for next token
                logits = self.output_proj(output[:, -1:, :])  # Only last position
                next_token = torch.argmax(logits, dim=-1)
                
                # Append to sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Check if all sequences have generated EOS token (assuming token id 2 is SEP)
                if torch.all(next_token == tokenizer.sep_token_id):  # SEP token
                    break
            
            return generated

    def generate_autoregressive(self, z,descriptor, max_length, batch_size, device, 
                           temperature=1.0, top_k=50, top_p=0.9):
        """
        Autoregressive generation with temperature, top-k, and top-p sampling
        for increased novelty in molecule SELFIES generation.
        
        Args:
            max_length: maximum generation length
            batch_size: number of sequences to generate
            device: torch device
            temperature: controls randomness (1.0 = original, >1.0 = more random)
            top_k: number of top tokens to consider for sampling
            top_p: cumulative probability threshold for nucleus sampling
        """
        # Start with CLS token
        generated = torch.ones(batch_size, 1, dtype=torch.long, device=device) * tokenizer.cls_token_id
        unfinished = torch.ones(batch_size, dtype=torch.bool, device=device)  # Track which sequences are still generating

        batch_size = z.size(0)
        device = z.device
        
        # Prepare condition (z + descriptor)
        descriptor = descriptor.unsqueeze(1).repeat(1, z.size(1), 1)
        condition = torch.cat([z, descriptor], dim=-1).to(self.z_proj.weight.dtype) 
        
        memory = self.z_proj(condition)  # [batch_size, seq_len, d_model]

        # Apply CNN layer
        # memory = condition.permute(0, 2, 1)  # Reshape to (batch_size, features, sequence_length)
        # memory = self.cnn_Decoder(memory.float())

        # memory = memory.permute(0, 2, 1)  # Reshape back to (batch_size, sequence_length, features)
        
        for _ in range(max_length - 1):
            # Only process unfinished sequences
            if not torch.any(unfinished):
                break
                
            token_emb = self.embed_tokens(generated)

            # Create causal mask
            seq_len = generated.size(1)
            tgt_mask = self.create_causal_mask(seq_len, device)
            
            # Transformer decoder
            output = self.transformer_decoder(
                tgt=token_emb,
                memory=memory,
                tgt_mask=tgt_mask
            )
            
            # Get logits for next token
            logits = self.output_proj(output[:, -1:, :])  # Only last position
            
            # Apply temperature scaling
            if temperature != 1.0:
                logits = logits / temperature
            
            # Apply top-k and top-p sampling
            next_token = self.sample_next_token(logits, top_k=top_k, top_p=top_p)
            
            # Only update unfinished sequences
            next_token[~unfinished] = tokenizer.pad_token_id  # Pad finished sequences
            
            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Update unfinished mask (stop when SEP token is generated)
            unfinished = unfinished & (next_token.squeeze(1) != tokenizer.sep_token_id)
        
        return generated

    def sample_next_token(self, logits, top_k=None, top_p=None):
        """
        Sample next token using temperature, top-k, and top-p sampling.
        
        Args:
            logits: raw model outputs [batch_size, 1, vocab_size]
            top_k: number of top tokens to consider
            top_p: cumulative probability threshold for nucleus sampling
        """
        logits = logits.squeeze(1)  # [batch_size, vocab_size]
        
        # Apply top-k filtering
        if top_k is not None and top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = -float('Inf')
        
        # Apply top-p (nucleus) sampling
        if top_p is not None and top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep the first token above threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            for idx in range(logits.size(0)):
                indices_to_remove = sorted_indices[idx][sorted_indices_to_remove[idx]]
                logits[idx][indices_to_remove] = -float('Inf')
        
        # Sample from the filtered distribution
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token


# In[107]:


# Updated AutoEncoder
class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.enc = encoder
        self.dec = decoder

    def forward(self, x, descriptor, desirable_descriptors, target_tokens=None):
        # import pdb
        # pdb.set_trace()
        enc_out = self.enc(x, descriptor)
        
        if target_tokens is not None:
            # Training: use teacher forcing
            dec_out = self.dec(enc_out, desirable_descriptors, target_tokens)
        else:
            # Inference: autoregressive generation
            dec_out = self.dec(enc_out, desirable_descriptors)
        
        return dec_out


# In[108]:


def autoregressive_loss(logits, targets, ignore_index=-100):
    """
    Compute cross-entropy loss for autoregressive generation
    
    Args:
        logits: [batch_size, seq_len, vocab_size]
        targets: [batch_size, seq_len]
    """
    # Shift logits and targets for next token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_targets = targets[..., 1:].contiguous()
    
    # Flatten for cross entropy
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_targets = shift_targets.view(-1)
    
    # Compute loss
    criterion = nn.CrossEntropyLoss(reduction='sum')
    loss = criterion(shift_logits, shift_targets)
    
    return loss


# In[109]:


class Lora_model(pl.LightningModule):
    def __init__(self, model, lr, vocab_size):
        super().__init__()
        self.model = model
        self.lr = lr
        self.vocab_size = vocab_size
        
        self.train_losses = []
        self.batch_losses = []
        self.similarities = []
        
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def forward(self, x, descriptor, desirable_descriptors, target_tokens=None):
        return self.model(x, descriptor, desirable_descriptors, target_tokens)
    
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = StepLR(optim, step_size=40, gamma=0.5)
        return [optim], [scheduler]

    def training_step(self, batch, batch_idx):
        tokens, canonical_smiles, descriptor, desirable_descriptors = batch
        
        # Forward pass with teacher forcing
        logits = self(tokens, descriptor, desirable_descriptors, target_tokens=tokens)
        
        # Compute reconstruction loss (autoregressive)
        self.recon_loss = autoregressive_loss(logits, tokens, ignore_index=tokenizer.convert_tokens_to_ids('[PAD]'))
        
        # Compute KL loss
        self.kl_loss = self.model.enc.kl_loss().sum(dim=[1, 2]).sum(dim=0)
        
        # # Total loss with beta annealing
        # if self.current_epoch > 9:
        #     beta = 0.1
        #     self.total_loss = self.recon_loss + beta * self.kl_loss
        # else:
        #     self.total_loss = self.recon_loss + self.kl_loss

        self.total_loss = self.recon_loss + self.kl_loss
        
        self.batch_losses.append(np.array([
            self.total_loss.data.item(),
            self.recon_loss.data.item(),
            self.kl_loss.data.item()
        ]))
        
        return {'loss': self.total_loss}

    def on_train_epoch_start(self):
        self.start = time.time()

    def on_train_epoch_end(self):
        self.train_losses.append(np.array(self.batch_losses).mean(axis=0))
        self.batch_losses = []
        
        print("Epoch time =", (time.time() - self.start)/60, "minute")
        print(f'Epoch {self.current_epoch} | Loss >> {self.train_losses[-1][0]:.4f}/ \
              {self.train_losses[-1][1]:.4f}/{self.train_losses[-1][2]:.4f}')
        
        with open("result_autoregressive_vae.txt", "a") as f:
            f.write('Epoch {} | Loss >> {:.4f}/{:.4f}/{:.4f}\n'.format(
                self.current_epoch, 
                self.train_losses[-1][0],
                self.train_losses[-1][1], 
                self.train_losses[-1][2]
            ))
            f.write('\n' + '_'*80 + '\n')

    def generate(self, descriptor, desirable_descriptors, z=None, max_length=256):
        """Generate molecules autoregressively"""
        self.eval()
        with torch.no_grad():
            if z is None:
                # Sample from prior
                batch_size = descriptor.size(0)
                z = torch.randn(batch_size, max_length, self.model.enc.z_size).to(descriptor.device)
            
            generated = self.model.dec(z, desirable_descriptors, max_length=max_length)
            return generated



# In[110]:


# Example usage and initialization
vocab_size=model.config.vocab_size
z_size=58

# num_desirable_descriptors=featuer_len

bert_encoder = BertEncoder(model)
encoder_var = EncoderVar(z_size, bert_encoder)

# Initialize autoregressive decoder
decoder = AutoregressiveDecoder(
    vocab_size=vocab_size,
    z_size=z_size,
    num_descriptors=num_desirable_descriptors,
    max_seq_len=150,
    d_model=768,
    nhead=12,
    num_layers=6,
    dropout=0.1)

# Create autoencoder
autoencoder = AutoEncoder(encoder_var, decoder)
    


# In[111]:


L_model = Lora_model(autoencoder, lr=0.0002, vocab_size=vocab_size)


# In[112]:


def count_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable_params, non_trainable_params



print(count_parameters(L_model))


# In[113]:


# Setup trainer
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath="checkpoints",
    filename="VEGFR2_autoregressive_desirable_desc_vae_{epoch}",
    verbose=True,
    every_n_epochs=1
)

trainer = pl.Trainer(
    accelerator="gpu",
    max_epochs=50,
    devices=[1],
    callbacks=[checkpoint_callback]
)


trainer.fit(model=L_model, train_dataloaders=train_loader
 ,ckpt_path=r"checkpoints/autoregressive_desirable_desc_vae_epoch=29.ckpt"
    )


# In[115]:


checkpoint = torch.load(r"checkpoints/VEGFR2_autoregressive_desirable_desc_vae_epoch=49.ckpt")
test_model =L_model
test_model.load_state_dict(checkpoint['state_dict'])


# In[116]:


device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
test_model.eval()
test_model.to(device)


# In[117]:


import re
import torch
from tqdm import tqdm

def remove_spaces_between_brackets(text):
    pattern = r"\]\s+|\s+\["
    return re.sub(pattern, lambda m: m.group().replace(' ', ''), text)

def decoded_strings_to_smiles(decoded_strings):
    """New function - converts already decoded strings to SMILES"""
    reconstructed_molecules = []
    for decoded_str in decoded_strings:
        cleaned_str = remove_spaces_between_brackets(decoded_str)
        try:
            smiles = sf.decoder(cleaned_str)
            reconstructed_molecules.append(smiles)
        except Exception as e:
            print(f"Error decoding '{cleaned_str}': {e}")
            reconstructed_molecules.append(None)  # or some default value
    return reconstructed_molecules


def calculate_similarities_autoregressive(test_model, train_loader, tokenizer, device="cuda"):
    test_model.eval()
    similarities = []
    print("Calculating similarities using full model…")

    with torch.no_grad():
        for (tokens, canonical_smiles, descriptor, desirable_descriptors) in tqdm(
            train_loader, total=len(train_loader), desc="Evaluating"
        ):
            # Move to device
            tokens = tokens.to(device)
            descriptor = descriptor.to(device)
            desirable_descriptors = desirable_descriptors.to(device)

            # 🔹 Run full model autoregressively
            reconstructed = test_model(
                tokens,
                descriptor,
                desirable_descriptors,
                target_tokens=tokens   # disable teacher forcing
            )  # [B, L]

            # Convert logits → token IDs
            reconstructed = torch.argmax(reconstructed, dim=-1)   # [B, L]

            # ---  Process generated sequences ----------------------------
            decoded_results = []
            for seq in reconstructed.tolist():
                # Truncate at EOS
                if tokenizer.sep_token_id in seq:
                    seq = seq[:seq.index(tokenizer.sep_token_id)]
                # Remove BOS if present
                if len(seq) > 0 and seq[0] == tokenizer.cls_token_id:
                    seq = seq[1:]
                decoded_str = tokenizer.decode(seq, skip_special_tokens=True)
                decoded_results.append(decoded_str)

            # ---  Compare original vs reconstructed ----------------------
            smi_result = decoded_strings_to_smiles(decoded_results)  # convert to SMILES

            for orig, recon in zip(canonical_smiles, smi_result):
                if recon is not None:  
                    sim = calculate_similarity(orig, recon, "Morgan")
                    if sim is not None:
                        similarities.append(sim)
                else:
                    print(f"Failed to decode molecule for: {orig}")
                if len(similarities)==155000:
                    print("avg Similarity",torch.tensor(similarities).mean())

                    

    # ---  Summary ------------------------------------------------------
    if similarities:
        avg = torch.tensor(similarities).mean()
        print(f"Average Similarity: {avg:.4f} (n={len(similarities)})")
    else:
        avg = torch.tensor(0.0)
        print("No valid similarities!")

    return avg, len(similarities)




# In[118]:


#Usage

# avg_sim,len_similarities= calculate_similarities_autoregressive(
#     test_model=test_model,
#     train_loader=train_loader,
#     tokenizer=tokenizer,
#     device=device
# )
# print(avg_sim)


#In[ ]:


# with open("result_autoregressive_vae_epoch_epoch=9.txt", "a") as f:
#     f.write('\ndesirable_desc_vae_epoch_=9')
#     f.write('\n___________________________________________\nlen of measured items  ')
#     f.write(str(len_similarities))
#     f.write('\ntrain Average similarity =')
#     f.write(str(avg_sim))


# In[121]:


tokens.size(1)

# In[ ]:
@torch.no_grad()
def generate_molecules_vae(model, tokenizer, descriptors_loader,
                           device="cuda", num_molecules=30000, max_len=150,
                           temperature=1.0, top_k=50, top_p=0.9):
    """
    Generate molecules from a trained VAE model using only the decoder
    with z ~ N(0, I), sampled once globally for all molecules.
    
    Args:
        model: Trained VAE model
        tokenizer: Tokenizer for encoding/decoding
        descriptors_loader: DataLoader for molecular descriptors
        device: Device to run on
        num_molecules: Number of molecules to generate
        max_len: Maximum sequence length
        temperature: Controls randomness (1.0 = original, >1.0 = more random)
        top_k: Number of top tokens to consider for sampling
        top_p: Cumulative probability threshold for nucleus sampling
    """
    model.eval()
    all_generated = []

    # 🔹 Sample all latent vectors once
    z_all = torch.randn(num_molecules, model.model.dec.z_size, device=device)

    idx = 0
    for batch in tqdm(descriptors_loader, desc="Generating molecules"):
        descriptors = batch
        descriptors = descriptors.to(device)
        batch_size = descriptors.size(0)

        # take the corresponding slice of z_all
        if idx + batch_size > num_molecules:
            batch_size = num_molecules - idx
            if batch_size <= 0:
                break
            descriptors = descriptors[:batch_size]

        z_batch = z_all[idx: idx + batch_size]   # [batch, z_size]
        idx += batch_size

        # expand z along sequence length
        z = z_batch.unsqueeze(1).repeat(1, max_len, 1)  # [batch, max_len, z_size]

        # Decode with sampling parameters
        generated_ids = model.model.dec.generate_autoregressive(
            z=z.to(device),
            descriptor=descriptors,
            max_length=max_len,
            batch_size=batch_size,
            device=device,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )

        # Convert ids -> SMILES strings
        for seq in generated_ids:
            seq = seq.tolist()
            if tokenizer.cls_token_id in seq:  # remove BOS
                seq = seq[seq.index(tokenizer.cls_token_id)+1:]

            if tokenizer.sep_token_id in seq:  # cut at EOS
                seq = seq[:seq.index(tokenizer.sep_token_id)]
            decoded = tokenizer.decode(seq, skip_special_tokens=True)
            all_generated.append(decoded)

            if len(all_generated) >= num_molecules:
                return decoded_strings_to_smiles(all_generated)

    return decoded_strings_to_smiles(all_generated)

avg_sim,len_similarities = calculate_similarities_autoregressive(
    test_model=test_model,
    train_loader=train_loader,
    tokenizer=tokenizer,
    device=device
)


# In[ ]:


print(avg_sim)


# In[ ]:


test_desirable_descriptors=pd.read_csv(r'VEGFR2/for_generation_VEGFR2_desirable_descriptors_30000.csv')


# # In[ ]:


test_desirable_descriptors = loaded_scaler.transform(test_desirable_descriptors)


# In[ ]:


with open("result_VEGFR2_autoregressive_vae.txt", "a") as f:
    f.write('\n___________________________________________len of measured items\n')
    f.write(str(len_similarities))
    f.write('\ntest Average similarity =')
    f.write(str(avg_sim))

# In[ ]:

from torch.utils.data import DataLoader, TensorDataset

def create_dataloader_from_numpy(array, batch_size=64, shuffle=True):
    data = torch.tensor(array, dtype=torch.float32)
    dataset = TensorDataset(data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

test_loader = create_dataloader_from_numpy(test_desirable_descriptors, batch_size=128)
# In[ ]:

generated_smiles = generate_molecules_vae(
    model=test_model,
    tokenizer=tokenizer,
    descriptors_loader=test_loader,  
    device=device,
    num_molecules=30000,
    max_len=150,
    temperature=1,  # More randomness
    top_k=100,        # Broader sampling
    top_p=1       # More diverse
)


print("generated_smiles",generated_smiles[:10])


# In[ ]:


# Create a DataFrame
df_gen = pd.DataFrame(generated_smiles, columns=["Smiles"])  # Use first row as header

# Write DataFrame to CSV
df_gen.to_csv(r"smiles_VEGFR2_autoregressive_desirable_desc_temp=1_epoch_49.csv", index=False)

print("********_______generated smiles saved______********")

print("number of unique generated smiles",len(set(generated_smiles)))

# In[ ]:

generated_smiles = generate_molecules_vae(
    model=test_model,
    tokenizer=tokenizer,
    descriptors_loader=test_loader,  
    device=device,
    num_molecules=30000,
    max_len=150,
    temperature=1.2,  # More randomness
    top_k=100,        # Broader sampling
    top_p=95       # More diverse
)


print("generated_smiles",generated_smiles[:10])


# In[ ]:


# Create a DataFrame
df_gen = pd.DataFrame(generated_smiles, columns=["Smiles"])  # Use first row as header

# Write DataFrame to CSV
df_gen.to_csv(r"smiles_sampling_VEGFR2_autoregressive_desirable_desc_epoch_49.csv", index=False)

print("********_______generated smiles saved______********")

print("number of unique generated smiles",len(set(generated_smiles)))