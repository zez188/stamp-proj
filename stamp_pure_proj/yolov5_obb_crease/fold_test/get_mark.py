import os
import cv2
import numpy as np
#打分脚本
'''
没有标签txt的图片，默认评分为0

'''
def get_mark():
    labels_path= 'exp/labels' #运行推理脚本，会在run/detect/exp/labels下产生标签
    list_file=os.listdir(labels_path)
    # list_file.sort()
    list_file_give=list_file
    if not isinstance(list_file_give,list):
        list_file_give=[list_file_give]
    mark_list=[]
    for filename in list_file_give:
        img_path=filename.replace('.txt','.jpg')
        original_img=cv2.imdecode(np.fromfile('exp/'+img_path,dtype=np.uint8),-1)
        row,col,c=original_img.shape
        txt_img=os.path.join(labels_path,filename)
        txt_fp=open(txt_img,'r')
        data=txt_fp.readlines()
        detect_num,length=0,0
        mark=1
        #阈值选取
        w_threshold,h_threshold=row*0.2,col*0.2
        for line in data:
            data_line=line.split()
            data_line=[float(s_data) for s_data in data_line]
            if data_line[0]==1:continue
            side_1=pow(pow(data_line[3]-data_line[1],2)+pow(data_line[4]-data_line[2],2),0.5)
            side_2=pow(pow(data_line[5]-data_line[1],2)+pow(data_line[6]-data_line[2],2),0.5)
            max_side=max(side_2,side_1)
            length+=max_side
            detect_num+=1
            for i in range(1,7,2):
                if data_line[i]>w_threshold and data_line[i]<col-w_threshold and data_line[i+1]>h_threshold and data_line[i+1]<row-h_threshold:
                    mark=2
                    continue
            if detect_num>5:mark=2
            if length>min(row,col)*0.2:mark=2
        mark_list.append(mark)
    #结果，list_file_give，mark_list
        print(img_path,mark)#打印图片以及对应分数.

get_mark()




















