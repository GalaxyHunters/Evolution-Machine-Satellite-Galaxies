from PIL import Image
import random
from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import scipy.stats as st

def df_stats(df):
    #sgal_features.info()
    print('# Rows: ', len(df), '# Columns: ', len(df.keys().values),
          df.keys().values)

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

#get_concat_h(im1, im2).save('data/dst/pillow_concat_h.jpg')
#get_concat_v(im1, im2).save('data/dst/pillow_concat_v.jpg')
