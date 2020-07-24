"""
This file is used to resize all the images to size 256x256 to standardise the images used for training

change the 'path' variable to point to the directory source containing the images
change the 'name' variable inside the last for loop to store the resized images into the target directory 
"""

from PIL import Image, ImageFilter, ImageEnhance, ImageFile
#import sys,os

'''
This file loads images from a given folder, creates a new set of augmented images,
and saves them in another folder.
'''
#from PIL import Image
import os, os.path

ImageFile.LOAD_TRUNCATED_IMAGES = True

imgs = []
newimg = []
name = []
path = '/home/ld07063u/new_images_chairs'
valid_images = [".jpg",".png",".tga", ".jpeg"]
for i,f in enumerate(os.listdir(path)):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    image = Image.open(os.path.join(path,f))
    rgb_im = image.convert('RGB')
    new_imgg = rgb_im.resize((256,256))
    print('iteration ',i)
    name = '/home/ld07063u/new_chairs_256x256/Image '+str(i+20000)+'.jpg'
    new_imgg.save(name, 'JPEG')


"""imgs.append(Image.open(os.path.join(path,f)))

for image in imgs:
    rgb_im = image.convert('RGB')
    newimg.append(rgb_im.resize((256,256)))

#it_list = [40000,50000,60000,70000,80000,90000,11000,12000,13000,14000,15000,16000,17000,18000,19000,20000,21000,22000,23000,24000,25000]

#for j in it_list:
for i, image in enumerate(newimg):
    print('iteration ',i)
    name = '/home/ld07063u/new_chairs_256x256/Image '+str(i+10000)+'.jpg'
    image.save(name, 'JPEG')"""