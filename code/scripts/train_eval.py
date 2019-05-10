import os
from random import randint
import shutil

path = "../data/finish-line/bmps/train/pos/"

def te(path):
    for filename in os.listdir(path):
        if filename.endswith(".bmp"):
            x = randint(0, 10)
            if x == 5:
                shutil.move(path + filename, path + "eval/" + filename)

path = "../data/finish-line/bmps/train/pos/"
te(path)
path = "../data/finish-line/bmps/train/neg/"
te(path)
