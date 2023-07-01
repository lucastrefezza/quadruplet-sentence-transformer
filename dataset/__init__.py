import os
import nltk
import random

import torch.cuda

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')

from dataset.constants import RANDOM_SEED

# Set random seed
random.seed(RANDOM_SEED)
torch.cuda.seed_all(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)