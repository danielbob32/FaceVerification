
#%%
import sys
import os

# Ensure the parent directory is in the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import numpy as np
import random
from matplotlib import pyplot as plt
from utils.helpers import transfer_lfw, uncompress_lfwa, setup_paths,acquire_data



#%% Setup paths for data
setup_paths()

#%% Uncompress lfw dataset
uncompress_lfwa()


#%% Transfer lfw into data/negative
transfer_lfw()

# %% acquire data
acquire_data()


# %%
