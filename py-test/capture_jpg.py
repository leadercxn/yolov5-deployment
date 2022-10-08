import os, random, shutil
import cv2 as cv
from tqdm import tqdm

def Resize_img(fileDir, tarDir):
    # print(os.path.exists(tarDir))
    # if not os.path.exists(tarDir) or os.path.exists(fileDir):  # 如果目标文件夹的子文件夹不存在，就在目标文件夹中建立子文件夹
    #     print("Dir Error!")
    #     return
    pathDir = os.listdir(fileDir)
    filenumber = len(pathDir)  # 计算文件总数
    print("Totle img",filenumber)
    for name in tqdm(pathDir):
        n,suffix = name.split(".")
        if not os.path.exists(tarDir):  # 如果目标文件夹的子文件夹不存在，就在目标文件夹中建立子文件夹
            os.mkdir(tarDir)
        img_path = os.path.join(fileDir, name)
        save_path = os.path.join(tarDir, n+".jpg")
        img = cv.imread(img_path)
        cv.imwrite(save_path,img)
    cv.destroyAllWindows()


if __name__ == '__main__':
    fileDir = r"./build"  # 源图片文件夹路径
    tarDir = r"./build"
    Resize_img(fileDir, tarDir)
    print("Extract Successfully!")
