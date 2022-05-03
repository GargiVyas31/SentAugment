from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from accelerate import Accelerator
import pandas as pd
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from transformers import default_data_collator
from tqdm import tqdm
import os
os.environ['HF_DATASETS_CACHE'] = '/work/gvyas_umass_edu/cache/'

accelerator = Accelerator()

def preprocess_function(examples):
    # Tokenize the texts
    texts = (examples["news_body"])
    result = tokenizer(texts, padding=True, max_length=128, truncation=True)
    return result

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", src_lang="en_XX")

dataset = pd.read_csv("data/master_training_data.csv", usecols = ["news_body", "news_category"]) #Or, I can give you a csv in the shared folder you can load from there.
dataset["news_body"][:100000].to_csv("data/data.csv", index=False)

data_files = {}
data_files["test"] = "data/data.csv"


raw_datasets = load_dataset("csv", data_files=data_files, cache_dir="/work/gvyas_umass_edu/cache/")
print(raw_datasets)
accelerator.wait_for_everyone()
with accelerator.main_process_first():
    processed_datasets = raw_datasets.map(
      preprocess_function, batched=True, desc="Running tokenizer on dataset" 
      )
#print(len(processed_datasets["test"]["news_body"]))

  

# # for key in split_data:
batch_size = 16
test_dataloader = DataLoader(
           processed_datasets["test"],
           collate_fn=default_data_collator,
           batch_size=batch_size
       )



model, test_dataloader = accelerator.prepare(model, test_dataloader)
translations = []
model.eval()

for step, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)/batch_size):
  generated_tokens = model.generate(**batch, forced_bos_token_id=tokenizer.lang_code_to_id["fr_XX"])
  translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
  translations.extend(translation)
  
#print(type(translations),len(translations), translations[0])
data = {"news_body": translations , "category":pd.read_csv("data/master_training_data.csv", usecols=["news_category"]).values.tolist()}
#print(data.head())
#print(type(data["category"]))
df = pd.DataFrame.from_dict(data)
df.to_csv('translated_data.csv', index=False)

# #Store translations in the column of datasets. Try w smaller sample first
