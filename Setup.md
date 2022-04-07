Create Conda environment.

`conda create -n sent_augment python=3.8`

`conda activate sent_augment`

Pytorch version for Unity cluster. Check right one [here](https://pytorch.org/).

`conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch`

Faiss download for GPU:

`conda install -c pytorch faiss-gpu`

Download sentencepiece.

`conda install -c conda-forge sentencepiece`

Downloading CC data. First access CPU resource.

`srun --time 2:00:00 --mem 40G --pty /usr/bin/bash`

Run script to download data.

`sbatch unity_scripts/download_cc_data.sh`

Download XML package inside SentAugment workspace.

`git clone https://github.com/facebookresearch/XLM`

Access a GPU instance for creating sentence embeddings.

`srun --time 1:00:00 --partition gpu --gres gpu:1 --mem 16G --exclude node41,node42,node92,node44 --pty /usr/bin/bash`

### For LASER:

Install dependencies. `path/to/model/directory` can be `/data`

`pip install laserembeddings`

`python -m laserembeddings download-models path/to/model/directory`

Encode bank sentences

`input=data/keys.txt`

`output=data/keys.pt`

`python src/laser.py --input $input --output $output --input_lang "en" --cuda "True" `

Encode input search sentences

`input=data/sentence.txt`

`python src/laser.py --input $input --output $input.pt --input_lang "fr" --cuda "True"`

Retrieve the Nearest Neighbors

`bank=data/keys.txt`

`emb=data/keys.pt`

`K=2`

`python src/flat_retrieve.py --input $input.pt --bank $bank --emb $emb --K $K > nn.txt &`

### For mDPR:

Download transformers library.

`conda install -c conda-forge transformers=4.16.2`

Embed bank sentences.

`input=data/keys_small.txt` (file should end with a newline)

`output=data/keys_small.pt`

`python src/mdpr.py --input $input --output $output --batch_size=4 --cuda "True" --load_saved "True"`

Embed input search sentences.

`input=data/sentence.txt`

`python src/mdpr.py --input $input --output $input.pt --batch_size=4 --cuda "True" --load_saved "True"`

Perform nearest neighbors search.

```
input=data/sentence.txt
bank=data/keys_small.txt
emb=data/keys_small.pt
K=2
output=data/sentences_nn.txt
python src/flat_retrieve.py --input $input --bank $bank --emb $emb --K $K --output $output
```


### For BM25:

Download dependency.

`pip install rank-bm25`

Tokenize bank/corpus, input and generate nearest neighbours. GPU is not needed.

`python src/bm25.py --input=data/sentence.txt --bank=data/keys_small.txt --K=3 --lowercase=True`

### Create MC4 dataset files:

Download dependencies.

```
pip install https://github.com/kpu/kenlm/archive/master.zip
conda install -c conda-forge datasets
```

Download Spacy for sentence tokenization.

```
pip install -U pip setuptools wheel
pip install -U spacy
python -m spacy download fr_core_news_md
python -m spacy download de_core_news_md
```


Create MC4 files for using as sentence bank. GPU is not needed.

```
file_name=data/mc4_fr10.txt
python src/generate_data.py --num_rows=10 --output $file_name --language=fr --split_by=sentence
```

For fast indexing, create a memory map of this file.

`python src/compress_text.py --input $file_name`
