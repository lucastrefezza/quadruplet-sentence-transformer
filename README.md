# Quadruplet Sentence Transformer

A research project on text-based information retrieval built around a simple idea: standard triplet-based training does not explicitly model **partial semantic inclusion** between texts.

This repository explores a novel training formulation, called **quadruplet loss**, which extends the standard anchor–positive–negative setup with a **partially positive example**. The goal is to fine-tune Sentence-BERT models so they better capture inclusion relationships between texts and descriptions, rather than only coarse similarity.

## Motivation

Many retrieval settings are not binary. A document may be:

- clearly relevant,
- clearly irrelevant,
- or only **partially relevant** because it captures a subset of the meaning of a more complete positive example.

Standard triplet loss does not model this middle case explicitly.  
This project introduces **quadruplet loss** to represent that structure during training.

A concrete example comes from **satellite image retrieval through captions**. Suppose the query describes a broad area, such as *"a coastal city with a harbor and surrounding residential districts"*. One image may match the full description, while another may depict only the harbor area. The second image is not fully irrelevant: it is **partially positive**, because its semantic content is included in the broader scene described by the query. Standard triplet loss treats relevance in a more rigid way, whereas quadruplet loss is designed to model this intermediate case explicitly.

## Main idea

The proposed loss extends the usual triplet formulation:

- **Anchor**
- **Positive example**
- **Negative example**

with a fourth element:

- **Partially positive example**, semantically included in the positive one

This makes it possible to train a sentence embedding model that is more sensitive to semantic inclusion relationships, which can be useful in information retrieval scenarios where relevance is graded rather than strictly binary.

## Project goals

- Fine-tune a Sentence-BERT model for information retrieval
- Model semantic inclusion between texts more explicitly
- Explore the usefulness of partially positive examples during training
- Compare the trained model against a baseline SentenceTransformer model

## Repository structure

```text
dataset/     Dataset creation, preprocessing, and example selection
models/      Model-related code
training/    Training logic
utils/       Utility functions
ir_evauation_script.py   Information retrieval evaluation script
part_pos_dataset.ipynb   Dataset exploration / construction notebook
quadruplet_loss_test.ipynb   Experimentation notebook
```

## Method overview

The pipeline includes:

dataset construction and preprocessing,
generation / selection of positive and partially positive examples,
training of a Sentence-BERT-based model with quadruplet loss,
information retrieval evaluation against a baseline model.

The evaluation setup is designed to compare a pretrained baseline with the trained model on IR metrics.

## Tech stack

Python
PyTorch
Sentence-Transformers
Transformers
Hugging Face Datasets
Weights & Biases

## Installation

Clone the repository and install dependencies:
```bash
git clone https://github.com/lucastrefezza/quadruplet-sentence-transformer.git
cd quadruplet-sentence-transformer
pip install -r requirements.txt
```
If you are using CUDA, you may prefer:
```bash
pip install -r requirements_cuda.txt
```

## Usage

The repository includes scripts and notebooks for:

- building datasets,
- selecting partially positive examples,
- training the model,
- running information retrieval evaluation.

A typical evaluation entry point is:

```bash
python ir_evauation_script.py --model_path trained/exp5
```

You can also configure:

- dataset paths,
- validation split,
- relevance settings for partially positive examples,
- baseline model,
- evaluation metrics and retrieval options.

## Research context

This project was developed as a research-oriented exploration of how sentence embedding models can be trained to better capture semantic inclusion, not just semantic similarity.

The central contribution is the design and testing of a quadruplet loss formulation for retrieval.

## Notes
The repository is experimental/research-oriented rather than packaged as a production library.
Some scripts and notebooks reflect exploration and iteration during development.
The current README is provided to not leave it blank, but it's only temporary and will be updated to provide a clearer overview of the project and its motivation as soon as possible.
