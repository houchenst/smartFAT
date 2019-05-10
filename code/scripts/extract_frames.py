import os
import shutil

written = open("extracted.txt", 'w')
path = "../data/finish-line/"
loc = path+"bmps/"
(total, used, free) = shutil.disk_usage(loc)
initial = used
for filename in os.listdir(path):
    (total, used, free) = shutil.disk_usage(loc)
    if filename.endswith("mp4") and used-initial < 25*1000000000:
        no_ex = filename[:-4]
        os.system("ffmpeg -i {0}{1} ../data/finish-line/bmps/{2}_$filename%03d.bmp".format(path, filename, no_ex))
        written.write(filename + "\n")