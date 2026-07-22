import glob
files = glob.glob('./Train_transliterated/*.xlsx')
# print(files)
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import namedtuple
from sklearn import preprocessing
# from bert import bert_tokenization
import numpy as np
import pandas as pd
df = pd.DataFrame()
from datasets import load_dataset
for file in files:
    df_new = pd.read_excel(file)
    columns = df_new.columns
    if "Code-Mix" not in columns:
        print(file)
    else:
        df = pd.concat([df,df_new],axis=0)
df = df[['Code-Mix', 'Emotion 1']] 
emotions= {
    "Serenity":0,
    "Annoyance":1,
    "Trust": 2,
    "Confusion":3,
    "Contempt":4,
    "Disgust":5,
    "Scared":6,
    "Sadness":7,
    "Surprise":8,
    "Joy":9,
    "Neutral":10,
    "Anger":11,
    "Anticipation":12,
}
print(df.columns)
print(df.shape)
df = df.dropna(subset=['Emotion 1','Code-Mix'])
df["label"] = df["Emotion 1"].map(emotions)
df["text"] = df["Code-Mix"]
df = df[["text",'label']]
df = df.astype(str)
df['label'] = df['label'].astype(int)
max_len = 512
from datasets import Dataset
dataset = Dataset.from_pandas(df,preserve_index=False)
print(dataset,dataset[0])
from transformers import AutoTokenizer
modelname = "ai4bharat/indic-bert"
tokenizer = AutoTokenizer.from_pretrained(modelname)
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=max_len, padding="max_length")
tokenized_datasets = dataset.map(tokenize_function, batched=True)
small_train_dataset, small_eval_dataset = tokenized_datasets.train_test_split(test_size=0.1).values()
small_test_dataset, small_eval_dataset = small_eval_dataset.train_test_split(test_size=0.5).values()
from transformers import AutoModelForSequenceClassification
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print(torch.cuda.get_device_name(0))
    device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
torch.cuda.empty_cache()
torch.cuda.device(0)
model = AutoModelForSequenceClassification.from_pretrained(modelname, num_labels=13).to(device)
from transformers import TrainingArguments
training_args = TrainingArguments(output_dir="test_trainer")
import numpy as np
from datasets import load_metric
import evaluate
# metric = load_metric("accuracy")
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
# metrics = [evaluate.load(metric_name) for metric_name in ["accuracy","f1","precision","recall"]]
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    # return metric.compute(predictions=predictions, references=labels)
    results = {}
    results.update({"accuracy":accuracy_score(predictions,labels)})
    results.update({"f1_macro":f1_score(predictions,labels,average='macro')})
    results.update({"f1_weighted":f1_score(predictions,labels,average='weighted')})
    results.update({"precision_macro":precision_score(predictions,labels,average='macro')})
    results.update({"precision_weighted":precision_score(predictions,labels,average='weighted')})
    results.update({"recall_macro":recall_score(predictions,labels,average='macro')})
    results.update({"recall_weighted":recall_score(predictions,labels,average='weighted')})
    return results
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
res = trainer.evaluate(eval_dataset=small_test_dataset)
print(res)
