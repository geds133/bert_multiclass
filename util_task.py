import pandas as pd
import torch
from tqdm.tk import tqdm

from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import time

start = time.time()
torch.cuda.is_available()
print(torch.version)
print(torch.version.cuda)
print(torch.backends.cudnn.enabled)


df = pd.read_csv(r'C:\Users\gerar\PycharmProjects\personal_projects\util\util_data\classification_dataset.csv\classification_dataset.csv')


df['classes'] = df.l1 + '_' + df.l2 + '_' + df.l3
df.classes.value_counts()

df['label'] = df.classes.rank(method='dense').astype(int)

X_train, X_val, y_train, y_val = train_test_split(df.text.values, df.label.values, test_size=0.15, random_state=42,
                                                  stratify=df.label.values)


# BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

encoded_data_train = tokenizer.batch_encode_plus(X_train, add_special_tokens=True, return_attention_mask=True,
                                                 padding=True, truncation=True, max_length=512, return_tensors='pt')

encoded_data_val = tokenizer.batch_encode_plus(X_val, add_special_tokens=True, return_attention_mask=True, padding=True,
                                               truncation=True, max_length=512, return_tensors='pt')

input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(y_train)

input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
labels_val = torch.tensor(y_val)

dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(df.label.unique()),
                                                      output_attentions=False, output_hidden_states=False)


batch_size = 10
dataloader_train = DataLoader(dataset_train, sampler=RandomSampler(dataset_train), batch_size=batch_size)
dataloader_validation = DataLoader(dataset_val, sampler=SequentialSampler(dataset_val), batch_size=batch_size)

optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-8)

epochs = 5
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(dataloader_train)*epochs)

from sklearn.metrics import f1_score
import numpy as np

def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')


def accuracy_per_class(preds, labels):

    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat == label]
        y_true = labels_flat[labels_flat == label]
        print(f'Accuracy: {len(y_preds[y_preds == label])}/{len(y_true)}\n')


seed_val = 17
torch.manual_seed(seed_val)
#torch.cuda.manual_seed_all(seed_val)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



def evaluate(dataloader_val):
    model.eval()

    loss_val_total = 0
    predictions, true_vals = [], []

    for batch in dataloader_val:
        batch = tuple(b.to(device) for b in batch)

        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

    loss_val_avg = loss_val_total / len(dataloader_val)

    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    return loss_val_avg, predictions, true_vals

Yt_train = Yt_train.type(torch.LongTensor)
Y_val = Yt_train.type(torch.LongTensor)

for epoch in range(epochs + 1):

    model.train()

    loss_train_total = 0

    for batch in dataloader_train:
        model.zero_grad()

        batch = tuple(b.to(device) for b in batch)

        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2],
                  }
        print('no error')
        outputs = model(**inputs)

        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

    torch.save(model.state_dict(), f'data_volume/finetuned_BERT_epoch_{epoch}.model')


    loss_train_avg = loss_train_total / len(dataloader_train)

    val_loss, predictions, true_vals = evaluate(dataloader_validation)
    val_f1 = f1_score_func(predictions, true_vals)



'''
# Reformer model.
from transformers import ReformerModel, ReformerConfig, ReformerTokenizer

# Initializing a Reformer configuration
configuration = ReformerConfig()

# Initializing a Reformer model
model = ReformerModel(configuration)

# Accessing the model configuration
configuration = model.config
'''