# trading

Strategy: put everything on 32 black

## Setup

### Virtual Environment (optional)
```
pip install virtualenvwrapper
mkdir ~/envs
echo "export WORKON_HOME=~/envs" >> ~/.bashrc
echo "source /usr/local/bin/virtualenvwrapper.sh" >> ~/.bashrc
mkvirtualenv trading
workon trading
```

### Getting the code
```git clone https://github.com/milokhl/trading.git```

### Dependencies
```pip install -r requirements.txt```


### Stanford OpenIE Wrapper
In the directory above ```/trading```
```
git clone https://github.com/milokhl/Stanford-OpenIE-Python.git
```

### Pre-trained Google News Word Vector Model
I used Google's pretrained word embedding model. It has a vocabulary of ~3 million words and the model is available here: <http://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/>
Download the pretrained binary file and put it in the directory above the project.

## Building the training set
The training pipeline takes in a bunch of financial news articles, does some intermediate processing, and eventually saves training data as numpy tensors.

### Download a financial news corpus
I used Philippe Rémy's financial news corpus: <https://github.com/philipperemy/financial-news-dataset>
It contains 450,341 news articles from Bloomberg and 109,110 news articles from Reuters and has been processed into a nice format.
Download as .zip and place in the directory above ```trading``` code.

### Extract events from the corpus
This is the most time consuming of the steps, so plan to run it overnight. We use Stanford OpenIE to extract event triples from all of the articles in the corpus, and write them to disk in batches.
```
python dataset.py extract path_to_corpus
```

### Build dictionaries
Now, we build dictionaries using all of the subjects, actions, and predicates found in our extracted event triples.
```
python build_dict --verbose --event_batch_size 400
```

### Write real/corrupt triples to disk
Using the dictionaries we created, we can save batches of real and corrupt triples to disk.
```
python write_pairs --verbose
```

### Write training tensors to disk
Finally, we convert our real and corrupt triples to their vector representation. Triples are stored in batches as numpy tensors.
```
python write_tensors --verbose --model path_to_googlenews_model --num_batches 20000 --batch_size 32
```
