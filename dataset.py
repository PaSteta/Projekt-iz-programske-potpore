import os
import os.path

import cv2

SHAPE_NET_RENDERING = "ShapeNetRendering"
SHAPE_NET_VOX32 = "ShapeNetVox32"
ROOT_SHAPE = "ShapeNet"
MAIN_DIR = f'./{ROOT_SHAPE}/{SHAPE_NET_RENDERING}'
MAIN_DIR_VOX = f'./{ROOT_SHAPE}/{SHAPE_NET_VOX32}'


def get_voxel_path(dir_name):
    obj_dirs = os.listdir(MAIN_DIR_VOX)

    obj_dir_found = ''

    for obj_dir in list(obj_dirs):
        d = f'{MAIN_DIR_VOX}/{obj_dir}'
        pic_dirs = os.listdir(d)
        if dir_name in pic_dirs:
            obj_dir_found = obj_dir
            break

    if obj_dir_found.__len__() == 0:
        print("Trazenje odgovarajuceg direktorija nije uspjelo!\n"
              "Direktorij ne postoji.")
        return ''

    print("Postoji vjerojatnost, ali jako mala, da postoje dva ili vise direktorija istog imena!\n"
          "Provjerite!")
    return f'{MAIN_DIR_VOX}/{obj_dir_found}/{dir_name}/model.binvox'


def get_object_data(obj_dir):
    data_list = []
    data = os.listdir(obj_dir)

    smanji_iter = 0             #defaultni br
    smanji_iter = len(data)*0.6 #samnjujem kolicinu podataka

    for pic_dir in range(int(len(data) - smanji_iter)):
        id = data[pic_dir]
        print(f'   loading dir {id} in {pic_dir} iteration...')

        if id.startswith('.'):
            continue

        shape_dir = f'{obj_dir}/{id}/rendering'  # Path do direktorija sa 24 slike

        # Učitavanje 24 slike u rječnik
        for pic in os.listdir(shape_dir):
            if '.png' in pic:  # If current item is a picture, store it
                img_array = cv2.imread(os.path.join(shape_dir, pic), cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (50, 50))  # resize to normalize data size
                data_list.append([new_array, id])

    return data_list


def get_train_data(object_num):
    if object_num > 13 or object_num < 1:
        return []
    data_list = []

    dir_iter = 0
    obj_dirs = os.listdir(MAIN_DIR)

    while dir_iter < object_num:
        d = obj_dirs[dir_iter]
        directory = f'{MAIN_DIR}/{d}'
        print("DIR:", d, "| iter:", dir_iter)
        obj_data = get_object_data(directory)
        data_list.extend(obj_data)
        dir_iter += 1

    return data_list
