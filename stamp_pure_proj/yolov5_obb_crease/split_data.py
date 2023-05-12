import os
import random

val_ratio=0.2


def split(files):
    txt_path='/data/ZGM/stamp_data/txt'
    train_path='/data/ZGM/mmrotate/data/train'
    val_path='/data/ZGM/mmrotate/data/val'
    test_path='/data/ZGM/mmrotate/data/test'
    img_root='/data/ZGM/stamp_data/fold_pick_img'

    val_img_path=os.path.join(val_path,'images')
    val_txt_ptah=os.path.join(val_path,'annfiles')
    train_img_path=os.path.join(train_path,'images')
    train_txt_ptah=os.path.join(train_path,'annfiles')

    os.system('rm -r '+train_txt_ptah+'/*')
    os.system('rm -r '+train_img_path+'/*')
    os.system('rm -r '+val_img_path+'/*')
    os.system('rm -r '+val_txt_ptah+'/*')

    for index,name in enumerate(files):
        img_cur_path=os.path.join(img_root,name).replace('.txt','.jpg')
        # os.system('mv '+img_cur_path+' '+img_cur_path.replace('.jpg','.png'))
        img_cur_path=img_cur_path.replace('.jpg','.png')

        if index<len(files)*0.2:
            os.system('cp '+img_cur_path+' '+val_img_path)
            os.system('cp '+os.path.join(txt_path,name)+' '+val_txt_ptah)
        else:
            os.system('cp '+img_cur_path+' '+train_img_path)
            os.system('cp '+os.path.join(txt_path,name)+' '+train_txt_ptah)

def pick_split_no_fold():
    img_root='/data/ZGM/yolov5_obb/dataset/val_split/images'
    txt_root='/data/ZGM/yolov5_obb/dataset/val_split/labelTxt'
    split_no_fold='/data/ZGM/yolov5_obb/dataset/split_no_fold'
    filenames=os.listdir(txt_root)
    for name in filenames:
        txt_path=os.path.join(txt_root,name)
        img_path=os.path.join(img_root,name).replace('.txt','.jpg')
        with open(txt_path,'r') as fw:
            data=fw.readlines()
            if not data:
                os.system('mv '+img_path+' '+split_no_fold)
                os.system('rm '+txt_path)
                print(name,'done')


def give_no_fold_to_train():
    split_no_fold='/data/ZGM/yolov5_obb/dataset/split_no_fold'
    train_img_path='/data/ZGM/yolov5_obb/dataset/val_split/images'
    filenames=os.listdir(split_no_fold)
    random.shuffle(filenames)
    for index,name in enumerate(filenames):
        img_path=os.path.join(split_no_fold,name)
        if index<100:
            os.system('cp '+img_path+' '+train_img_path)
        else:
            break








if __name__ == '__main__':
    # txt_path='/data/ZGM/stamp_data/txt'
    # filenames=os.listdir(txt_path)
    # random.shuffle(filenames)
    # split(filenames)
    pick_split_no_fold()
    give_no_fold_to_train()