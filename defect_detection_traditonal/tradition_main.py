import cv2
import numpy as np
import myimage_pre as mp
import os
from tqdm import tqdm


def check1image(image_path, min_area, blocksize=21, C=7, cvt2rgb=False,threshold_val=None):
    # 1. 读取图片
    image = mp.read_image_cv2(image_path=image_path, cvt2rgb=cvt2rgb)


    # 创建一个用于绘制结果的副本
    output_image = image.copy()
    color_change = 'cv2.COLOR_BGR2GRAY'

    if cvt2rgb:
        color_change = 'cv2.COLOR_RGB2GRAY'

    gray = cv2.cvtColor(image, eval(color_change))
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. 二值化处理
    
    #_, thresh = cv2.threshold(blurred, threshold_val, 255, cv2.THRESH_BINARY_INV)

    # 关键修改：使用 cv2.THRESH_OTSU 自动计算阈值
    #_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # 假设 blurred 是你的高斯模糊图像
    thresh = cv2.adaptiveThreshold(blurred, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  cv2.THRESH_BINARY_INV,blocksize, C)
    #_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    

    # 4. 形态学操作（闭运算）以填充轮廓中的孔洞
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    #closed_inverted = cv2.bitwise_not(closed)

    # 5. 查找轮廓
    # cv2.RETR_EXTERNAL 只检测外轮廓
    # cv2.CHAIN_APPROX_SIMPLE 压缩轮廓点
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    


    # 6. 过滤并绘制轮廓
    contours_filtered = []
    for contour in contours:
        # 过滤掉面积过小的轮廓
        if cv2.contourArea(contour) > min_area:
            cv2.drawContours(output_image, [contour], -1, (255, 0, 0), 1)
            contours_filtered.append(contour)
    
    #return image, gray, blurred, thresh,closed,output_image, contours,contours_filtered
    return output_image,contours_filtered




def work(images_folder_read, images_folder_save, cvt2rgb=False,blocksize=31, min_area=5):
    '''
    folder: image to read and save
    cvt2rgb: whether to change
    
    '''
    #go through all pictures
    for fp in tqdm(os.listdir(images_folder_read)):
        pic_temp = os.path.join(images_folder_read, fp)
        print(pic_temp)
        
        output_image,contours_filtered = check1image(image_path=pic_temp, min_area=min_area,blocksize=blocksize)
        
        f_save = f'{images_folder_save}_0/{fp}'
        if len(contours_filtered)>0:
            f_save = f'{images_folder_save}_1/{fp}'
            if cvt2rgb:
                output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        

        cv2.imwrite(f_save, output_image)


if __name__ == '__main__':
    images_folder_read =  'C:/Drivers/01_code_template/010_computer_vision/02_case_defect_detection/02_pictures/Images0001'
    images_folder_save = 'C:/Drivers/01_code_template/010_computer_vision/02_case_defect_detection/check_result'
    #work(images_folder_read=images_folder_read, images_folder_save=images_folder_save, cvt2rgb=False,blocksize=31, min_area=5)
    work(images_folder_read,images_folder_save)

