import cv2
import os
import random
import numpy as np
from tqdm import tqdm
def perform_aug(img,aug_type):
    if aug_type=="mdb":
        return cv2.medianBlur(img,5)
    elif aug_type=="gb":
        return cv2.GaussianBlur(img,(5,5),0)
    elif aug_type=="flt2D":
        kernel=np.ones((5,5),np.float32)/25
        return cv2.filter2D(img,-1,kernel)
    elif aug_type=="flv":
        return cv2.flip(img,1)
    elif aug_type=="flh":
        return cv2.flip(img,0)
    
def augment_files(folder_path,number_of_ops):
    aug_ops=['mdb','gb','flt2D','flh','flv']
    files=os.listdir(folder_path)
    no_of_files=len(files)
    print(str(number_of_ops)+" to perform on "+str(no_of_files)+" file(s)")
    print("Starting...")
    for i in tqdm(range(0,number_of_ops-1)):
        #pick a random file
        rand_index_file=random.randint(0,no_of_files-1)
        file_path=os.path.join(folder_path,files[rand_index_file])
        rand_aug=random.randint(0,len(aug_ops)-1)
        img=cv2.imread(file_path)
        transformed_img=perform_aug(img,aug_ops[rand_aug])
        split_filename=files[rand_index_file].split('.')
        file_title=split_filename[-3]+".aug"+split_filename[-2]+".jpg"
        save_file_path=os.path.join(folder_path,file_title)
        #print save_file_path
        cv2.imwrite(save_file_path,transformed_img)
    print("Done")
augment_files('E:\\python_stuffs\\catdog\\train_data\\train',10000)
