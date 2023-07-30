import transformers
from transformers import GPT2LMHeadModel, AutoTokenizer
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

#######################################################################################################################

# Generating text from the base model
# Importing the gpt2 model
tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

tokens = tokenizer(['Any input text'], return_tensors='pt')
print('tokens:', tokens)
print(tokenizer.batch_decode(tokens['input_ids']))

output = model.generate(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'], max_length=30)
print('output:', output)
print(tokenizer.batch_decode(output))

#######################################################################################################################

# Altering the configs
config = transformers.GPT2Config.from_pretrained('gpt2')
#print(config)
config.do_sample = config.task_specific_params['text-generation']['do_sample']
config.max_length = config.task_specific_params['text-generation']['max_length']

# Creating a pre-trained model defining new config
model = GPT2LMHeadModel.from_pretrained('gpt2', config=config)

#### Manually generating text from logits using a loop
input_ids = torch.tensor([[300, 400, 500]]) # any integer corresponding to a token in the vocab
attention_mask = torch.tensor([[1, 1, 1]])
output = model(input_ids=input_ids, attention_mask=attention_mask)
print(output.logits)
print(output.logits.shape)

token_1 = torch.tensor([464])
# Each logit is a predicted token (next word that was predictid)
token_2 = torch.argmax(output.logits[0][0], keepdim=True)
token_3 = torch.argmax(output.logits[0][1], keepdim=True)
token_4 = torch.argmax(output.logits[0][2], keepdim=True)
print(token_1, token_2, token_3, token_4)

print(tokenizer.batch_decode(token_1))
#
print(tokenizer.batch_decode(token_2))
print(tokenizer.batch_decode(token_3))
print(tokenizer.batch_decode(token_4))

# Logits together
predictions_ = torch.argmax(output.logits[0], dim=-1, keepdim=True)
print('\n', predictions_, '\n', tokenizer.batch_decode(predictions_))

###
input_ids = tokens['input_ids']
for i in range(20):
  attention_mask = torch.ones(input_ids.shape, dtype=torch.int64)
  logits = model(input_ids=input_ids, attention_mask=attention_mask)['logits']
  new_id = logits[:, -1, :].argmax(dim=1) # Generate new id
  #new_id = torch.randint(0, len(logits[0,0,:]), (1,)) # Random token generation
  input_ids = torch.cat([input_ids, new_id.unsqueeze(0)], dim=1)

print(tokenizer.batch_decode(input_ids))

#######################################################################################################################

# Giving any text data we want our model to be fine-tuned
data = ['My name is Any_Name.',
        'Any_name is 25 years old. He works at Company_name',
        'Another_name is a real boss. Bu he does not share his secrets how to become Any_position',
        'Superviser_name is the supervisor of previously_defined_name']

tokens = tokenizer(data)

class MyCustomDataset(Dataset):
  def __init__(self, input_data):
    self.data = []
    for i, j in zip(input_data['input_ids'], input_data['attention_mask']):
      input_ids = torch.tensor(i, dtype=torch.int64)
      attention_mask = torch.tensor(j, dtype=torch.int64)
      self.data.append({'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': input_ids})

  def __getitem__(self, index):
    return self.data[index]

  def __len__(self):
    return len(self.data)

dataset = MyCustomDataset(tokens)
data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

COST = []
model.train()
for epoch in range(5):
  LOSS = 0
  for batch in data_loader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs['loss']
    # logits = outputs.logits
    # loss = criterion(logits, labels)
    LOSS += loss.item()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

  COST.append(LOSS)

model.save_pretrained('fine_tuned_gpt2')

### Loading the model followed by generating text
model = GPT2LMHeadModel.from_pretrained('fine_tuned_gpt2')
tokenizer = AutoTokenizer.from_pretrained('gpt2')

input_tokens = tokenizer('Supervisors_name', return_tensors='pt')
output = model.generate(input_ids=input_tokens['input_ids'], attention_mask=input_tokens['attention_mask'], max_length=10)
tokenizer.batch_decode(output)
