import os
import sys
import nltk
import random
import numpy as np
import torch
from dataset.constants import RANDOM_SEED

# Init NLTK additional dependencies
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
sys.stderr.flush()
sys.stdout.flush()

# Set random seed
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)