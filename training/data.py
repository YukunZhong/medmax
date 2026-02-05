import torch
import string
import jsonlines
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

# Define the dataset class
class TokenizedDataset(Dataset):
    def __init__(self, filepath, max_length=2048):
        self.tokenized_data = []
        with jsonlines.open(filepath) as reader:
            for obj in reader:
                if 'tokens' not in obj:
                    raise KeyError("Missing required 'tokens' field in input data.")
                self.tokenized_data.append(torch.tensor(obj['tokens'], dtype=torch.long)[:max_length])

    def __len__(self):
        return len(self.tokenized_data)
    
    def __getitem__(self, idx):
        return self.tokenized_data[idx],

# Define custom collate function for DataLoader
def collate_fn(batch):
    batch_inputs = [item[0] for item in batch]
    ## padding value = 1 since tokenizer.pad_token_id = 1 for Chameleon/Anole
    batch_inputs_padded = pad_sequence(batch_inputs, batch_first=True, padding_value=1)

    labels = batch_inputs.copy()
    location_sep = (labels[0] == 8710).nonzero()
    ## if there are pad tokens
    if len(location_sep) != 0:
        new_labels = []
        for i in range(len(labels)):
            label = labels[i]
            loc_sep = (label == 8710).nonzero()[0]
            new_label = torch.cat([torch.tensor([-100] * (loc_sep+1)), label[loc_sep+1:]], dim = -1)
            new_labels.append(new_label)
        labels = new_labels
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    # Create attention masks
    attention_masks = torch.zeros_like(batch_inputs_padded, dtype=torch.long)
    attention_masks = attention_masks.masked_fill(batch_inputs_padded != 1, 1)
   
    return {'input_ids': batch_inputs_padded, 'attention_mask': attention_masks, 'labels': labels}

def create_new_tokens(extended_vocab_size):
    ## assuming extended_vocab size <= 26 x 26 x 26
    prefix = "IMGIMGIBJBZ{X}{Y}{Z}"
    alphabets = string.ascii_uppercase
    new_tokens = []
    c = 0
    for x in alphabets:
        for y in alphabets:
            for z in alphabets:
                new_tokens.append(prefix.format(X=x, Y=y, Z=z))    
                c += 1   
                if c == extended_vocab_size:
                    return new_tokens