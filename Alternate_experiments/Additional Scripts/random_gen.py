# Sample a random image from each sub-directory
# Add this random image to a particular folder

import os 
import shutil
import glob
import random

import matplotlib.pyplot as plt
import matplotlib.image as img

from tqdm import tqdm 
  


def imdb_wiki_randomizer(path, target):
    image_list = []
    dir_name_list = ['00','01','02','03','04','05','06','07','08','09']

    for elem in range(10,100):
        dir_name_list.append(str(elem))


    add_p = random.choice(dir_name_list)
    add_p = "/" + add_p
    

    path = path + add_p

    random_filename = random.choice([
        x for x in os.listdir(path)
        if os.path.isfile(os.path.join(path, x))
    ])

    original = path + "/" + random_filename
    

    shutil.copy(original, target)

    # for filename in glob.glob('C:/Users/nsbha/gitCode/age-gender-estimation/data/imdb_crop')

for i in tqdm(range(100), desc = "Copying..."):

    imdb_wiki_randomizer("C:/Users/nsbha/gitCode/age-gender-estimation/data/imdb_crop", "C:/Users/nsbha/Desktop/17th_August_Results/RDM_Images_IMDB_2")