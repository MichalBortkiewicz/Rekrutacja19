# -*- coding: utf-8 -*-

import numpy as np
import os
import pandas as pd
import cv2

def calc_distance(x,y):
    return (x**2 + y**2)**1/2

#%%
img_names = os.listdir("images")

imgs_paths = [os.path.join("images", item) for item in img_names]

images = [cv2.imread(item_path, cv2.IMREAD_ANYCOLOR) for item_path in imgs_paths]

img_info_df = pd.DataFrame(columns = ['nazwa_pliku', 'opis_zawar_zdj',
                                      'id_zdj', 'szer', 'wys', 'sr_kolor',
                                      'med_jasn_gray', 'x_max_gray', 'y_max_gray'])
img_info_df['nazwa_pliku'] = img_names

#%%
img_names_splitted = [elem.split("-") for elem in img_names]

img_opis = [' '.join(elem[2:len(elem)-1]) for elem in img_names_splitted]

img_info_df['opis_zawar_zdj'] = img_opis

#%%
img_info_df['id_zdj'] = [elem[len(elem)-1][:len(elem[len(elem)-1])-4]
                        for elem in img_names_splitted]

#%%
img_info_df['szer'] = [img.shape[1] for img in images]
img_info_df['wys'] = [img.shape[0] for img in images]

#%%
img_info_df['sr_kolor'] = [(np.mean(img[:,:,0]),
           np.mean(img[:,:,1]),
           np.mean(img[:,:,2])) for img in images]

#%%
gray_imgs = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]
img_info_df['med_jasn_gray'] = [np.median(img) for img in gray_imgs]


#%%
indicies = [list(np.where(img == img.max())) for img in gray_imgs]

x_max_gray = []
y_max_gray = []

for i in range(len(indicies)):
    min_dist = calc_distance(indicies[i][0][0], indicies[i][1][0])
    y_max = indicies[i][0][0]
    x_max = indicies[i][1][0]
    if (len(indicies[i][0]) > 1):
        for j in range(1, len(indicies[i][0])):
            dist = calc_distance(indicies[i][0][j], indicies[i][1][j])
            if dist < min_dist:
                min_dist = dist
                y_max = indicies[i][0][j]
                x_max = indicies[i][1][j]
    x_max_gray.append(x_max)
    y_max_gray.append(y_max)

#%%
img_info_df['x_max_gray'] = x_max_gray
img_info_df['y_max_gray'] = y_max_gray

#%%
img_info_df.to_csv("images.csv", index=False)

#%% agregacja
img_info_df_sorted = img_info_df.sort_values(by = "med_jasn_gray")
img_names_sorted = img_info_df_sorted["nazwa_pliku"]

kubelki_path = "agg-images"
os.makedirs(kubelki_path, exist_ok=True)

kubelek = 0
for i, img_name in enumerate(img_names_sorted):
    if i % 4 == 0:
        kubelek+=1
        img_dest_dir = os.path.join(kubelki_path, str(kubelek) + "-images")
        os.makedirs(img_dest_dir, exist_ok=True)

    img_src_dir = "images"
    os.rename(os.path.join(img_src_dir, img_name), os.path.join(img_dest_dir, img_name))

