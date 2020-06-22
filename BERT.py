def separate(params):
  print("\n\n----------------------------------- {} ---------------------------------------".format(params))

import pandas as pd
# pandas documentation
# https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html

separate("LOAD TSV FILE")
# Load the dataset into a pandas dataframe.
df = pd.read_csv("./cola_public/raw/in_domain_train.tsv", delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])

# todo Report the number of sentences
# @link https://stackoverflow.com/questions/10200268/what-does-shape-do-in-for-i-in-rangey-shape0/21200291#:~:text=shape%20is%20a%20tuple%20that%20gives%20dimensions%20of%20the%20array..&text=shape%20is%20a%20tuple%20that%20gives%20you%20an%20indication%20of,first%20dimension%20of%20your%20array.
# shape is a tuple that gives dimensions of the array.
# .shape[0] -> give row's count
# .shape[1] -> give column's count
print('Number of training sentences: {:,}'.format(df.shape[0]))
print('Number of column: {:,}'.format(df.shape[1]))

# todo display 10 random rows from the data.
separate("10 Rows")
df.sample(10)

# 4936	ks08	1	NaN	Mary, who John asked for help, thinks he is fo...
# 7446	sks13	0	*	Mary thinks for Bill to come.
# 363	bc01	1	NaN	Mary hired someone.
# 4059	ks08	0	*	She pinched that he feels pain.
# 4519	ks08	1	NaN	You can do it, but you better not.
# 2552	l-93	0	*	Cheryl stood the shelf with the books.
# 3575	ks08	1	NaN	Students wanted to write a letter.
# 3986	ks08	0	*	Talked with Bill about the exam.
# 3661	ks08	1	NaN	John seems certain about the bananas.
# 6655	m_02	1	NaN	Vera is knitting there.

# todo find label = 0 and get 5 rows, just column sentence and label
# .loc is panda
df.loc[df.label == 0].sample(5)[['sentence', 'label']]
#                               sentence	    label
# 5896	John placed the flute.	                0
# 6239	Is likely Jean to leave.	            0
# 2620	Doug removed the scratches to nowhere.	0
# 3892	John placed him busy.	                0
# 7054	I have gone and buys some whiskey.	    0

# Get the lists of sentences and their labels.
sentences = df.sentence.values
print("sentences : \n", sentences)
labels = df.label.values
print("labels : \n", labels)


# Tokenization & Input Formatting
# BERT Tokenizer
# from transformers import BertTokenizer
import transformers

# Load the BERT tokenizer.
separate("LOAD BERT TOKENIZER")
print('Loading BERT tokenizer...')
print('Using "bert-base-uncased"')
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
# what is 'bert-base-uncased' ?
# https://huggingface.co/transformers/pretrained_models.html

# apply the tokenizer to one sentence just to see the output.
# Print the original sentence.
separate("TEST APPLY TOKENIZER TO ONE SENTENCE")
print('Original: ', sentences[0])

# Print the sentence split into tokens.
print('Tokenized: ', tokenizer.tokenize(sentences[0]))

# Print the sentence mapped to token ids.
print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentences[0])))

# Original:  Our friends won't buy this analysis, let alone the next one we propose.
# Tokenized:  ['our', 'friends', 'won', "'", 't', 'buy', 'this', 'analysis', ',', 'let', 'alone', 'the', 'next', 'one', 'we', 'propose', '.']
# Token IDs:  [2256, 2814, 2180, 1005, 1056, 4965, 2023, 4106, 1010, 2292, 2894, 1996, 2279, 2028, 2057, 16599, 1012]

# setiap mengubah kata menjadi token ID, kita tinggal memanggil function tokenize.encode
# data yang dipakai akan ditambahin dengan token pada awal dan akhir kalimat.
# [CLS] kalimat yang sangat banyak [SEP]
separate("ALL SENTENCE DATA")
max_len = 0

print ('sentences : \n',sentences)
# For every sentence...
for sent in sentences:
    # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
    input_ids = tokenizer.encode(sent, add_special_tokens=True)
    # print(sent," <> ", input_ids)
    # Update the maximum sentence length.
    max_len = max(max_len, len(input_ids))

print('Max sentence length: ', max_len)


import torch
# Tokenize all of the sentences and map the tokens to thier word IDs.
input_ids = []
attention_masks = []

# For every sentence...
for sent in sentences:
    # `encode_plus` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    #   (5) Pad or truncate the sentence to `max_length`
    #   (6) Create attention masks for [PAD] tokens.
    encoded_dict = tokenizer.encode_plus(
        sent,  # Sentence to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=64,  # Pad & truncate all sentences.
        pad_to_max_length=True,
        return_attention_mask=True,  # Construct attn. masks.
        return_tensors='pt',  # Return pytorch tensors.
    )

    # Add the encoded sentence to the list.
    input_ids.append(encoded_dict['input_ids'])

    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])

# Convert the lists into tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

# Print sentence 0, now as a list of IDs.
print('Original: ', sentences[0])
print('Token IDs:', input_ids[0])

from torch.utils.data import TensorDataset, random_split

# Combine the training inputs into a TensorDataset.
dataset = TensorDataset(input_ids, attention_masks, labels)

# Create a 90-10 train-validation split.

# Calculate the number of samples to include in each set.
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

# Divide the dataset by randomly selecting samples.
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))


from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

# The DataLoader needs to know our batch size for training, so we specify it
# here. For fine-tuning BERT on a specific task, the authors recommend a batch
# size of 16 or 32.
batch_size = 32

# Create the DataLoaders for our training and validation sets.
# We'll take training samples in random order.
train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )


separate("LOAD BertForSequenceClassification")
from transformers import BertForSequenceClassification, AdamW, BertConfig

# Load BertForSequenceClassification, the pretrained BERT model with a single
# linear classification layer on top.
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 2, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)

# If there's a GPU available...
if torch.cuda.is_available():
    print("::RUN WITH CUDA")
    model.cuda()
else:
    print("::RUN WITH CPU")
    model.cpu()

# Tell pytorch to run this model on the GPU.
