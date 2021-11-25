import pandas as pd
import torch
from tqdm.notebook import tqdm

from transformers import BertTokenizer
from torch.utils.data import TensorDataset

from transformers import BertForSequenceClassification

df = pd.read_csv(r'C:\Users\gerar\PycharmProjects\personal_projects\util\util_data\classification_dataset.csv\classification_dataset.csv')

[print(col, df[col].unique()) for col in df.columns[1:]]

df['classes'] = df.l1 + '_' + df.l2 + '_' + df.l3
df.classes.value_counts()