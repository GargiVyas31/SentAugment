Create Conda environment.

`conda create -n sent_augment python=3.8`

`conda activate sent_augment`

Pytorch version for Unity cluster. Check right one [here](https://pytorch.org/).

`conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch`

Faiss download for GPU:

`conda install -c conda-forge faiss-gpu`

Download sentencepiece.

`conda install -c conda-forge sentencepiece`

Downloading CC data. First access CPU resource.

`srun --time 2:00:00 --mem 40G --pty /usr/bin/bash`

Run script to download data.

`sbatch unity_scripts/download_cc_data.sh`

Download XML package inside SentAugment workspace.

`git clone https://github.com/facebookresearch/XLM`

Access a GPU instance for creating sentence embeddings.

`srun --time 1:00:00 --partition gpu --gres gpu:1 --mem 100G --pty /usr/bin/bash`

#### For LASER:

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

#### For mDPR:

Download transformers library.

`conda install -c conda-forge transformers=4.12.2`

Embed bank sentences.

`input=data/keys_small.txt` (file should end with a newline)

`output=data/keys_small.pt`

`python src/mdpr.py --input $input --output $output --batch_size=4 --cuda "True"`

Embed input search sentences.

`input=data/sentence.txt`

`python src/mdpr.py --input $input --output $input.pt --batch_size=4 --cuda "True"`

Perform nearest neighbors search.

`input=data/sentence.txt`

`bank=data/keys_small.txt`

`emb=data/keys_small.pt`

`K=2`

`python src/flat_retrieve.py --input $input.pt --bank $bank --emb $emb --K $K > nn.txt`
