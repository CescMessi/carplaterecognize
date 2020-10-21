import cv2
import numpy as np
from PIL import Image
from findplate.testnetwork import detect
from findplate.testnetwork import identify


# 图像预处理
def preprocess(img):
    # 将图片转换为HSV颜色空间
    hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    # 车牌照为蓝色，设置蓝色的hsv阈值，提取出图片中的蓝色区域
    h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]
    plate_color_img = (((h > 100) & (h < 124))) & (s > 120) & (v > 60)
    # 将图片数据格式转为8UC1的二值图
    plate_color_img = plate_color_img.astype('uint8') * 255
    # 对图片进行膨胀处理，使车牌成为一个整体
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    plate_color_img = cv2.dilate(plate_color_img, element, iterations = 1)
    return plate_color_img

# 找到车牌位置
def findPlate(plate_color_img, im):
    # 在膨胀后的二值图像中寻找所有的轮廓，并存入数组
    contours, hierarchy = cv2.findContours(plate_color_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    # 遍历轮廓
    for contour in contours:
        area = cv2.contourArea(contour)
        # 去除面积很小的轮廓
        if (area < (1/500 * plate_color_img.shape[0] * plate_color_img.shape[1]) ):
            continue
        
        # 获取轮廓的最小外接矩形
        rect = cv2.minAreaRect(contour)
        rect_point = cv2.boxPoints(rect)
        rect_point = np.int0(rect_point)
        
        # 将矩形顶点重新排序，左上角开始顺时针排序
        k = 0
        min_point = rect_point[0][0] + rect_point[0][1]
        for i in range(len(rect_point)):
            if (rect_point[i][0] + rect_point[i][1] < min_point):
                min_point = rect_point[i][0] + rect_point[i][1]
                k = i

        new_rect = [rect_point[k], rect_point[(k+1)%4], rect_point[(k+2)%4], rect_point[(k+3)%4]]
        
        # 通过仿射变换对车牌图片进行校正，存入新图像
        plate_img = np.zeros((140,440,3), np.uint8)
        pts1 = np.float32(new_rect)
        pts2 = np.float32([[0,0],[440,0],[440,140],[0,140]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        plate_img = cv2.warpPerspective(im, matrix, (440,140))

        # 将图像转为PIL图像，喂入神经网络检测该区域是否为车牌
        detect_image = Image.fromarray(cv2.cvtColor(plate_img,cv2.COLOR_BGR2RGB))
        result = detect(detect_image)
        print(result)
        if (result[0] == 'has'):
            return rect_point, plate_img
    
    return rect_point, plate_img

# 拆分字符
def getChar(plate_binary):
    plate_height, plate_width = plate_binary.shape[:2]
    
    # 将二值图像中像素投影到y轴计数
    y_white_pixels = [0 for x in range(plate_height)]
    for i in range(plate_height):
        for j in range(plate_width):
            if (plate_binary[i,j] == 255):
                y_white_pixels[i] += 1

        # 通过占行像素的比例去除边框和杂质
        if (y_white_pixels[i] < 0.1*plate_width or y_white_pixels[i] > 0.8*plate_width ):
            y_white_pixels[i] = 0
    

    # 选取最长的投影作为字符位置
    flag = 0
    index = 0
    y_lenth = 0
    y_white_list = []
    for i in range(plate_height):
        if y_white_pixels[i] != 0:
            if flag == 0:
                index = i
                flag = 1
            y_lenth += 1
        elif flag == 1:
            flag = 0
            y_white_list.append([index, y_lenth])
            y_lenth = 0
    y_white_list.sort(key=lambda x:x[1], reverse=True)
    y_top = y_white_list[0][0]
    y_bottom = y_top + y_white_list[0][1] - 1
    y_crop_img = plate_binary[y_top:y_bottom, :]
    cv2.imshow('yci',y_crop_img)
    # cv2.waitKey()

    # 将像素对x轴投影，选取最长的7个投影
    x_white_pixels = [0 for x in range(plate_width)]
    for i in range(plate_width):
        for j in range(y_crop_img.shape[0]):
            if (y_crop_img[j,i] == 255):
                x_white_pixels[i] += 1
    
    flag = 0
    index = 0
    x_lenth = 0
    x_white_list = []
    for i in range(plate_width):
        if x_white_pixels[i] >= 6:
            if flag == 0:
                index = i
                flag = 1
            x_lenth += 1
            # 添加图像边缘的投影
            if i == plate_width - 1:
                x_white_list.append([index, x_lenth])
        elif flag == 1:
            flag = 0
            x_white_list.append([index, x_lenth])
            x_lenth = 0
    print(x_white_list)

    # 去除中间的点
    for x in x_white_list:
        flag = 0
        if x[1] < 20:
            for i in range(x[1]):
                if x_white_pixels[x[0]+i] > 0.5 * y_crop_img.shape[0]:
                    flag = 1
                    break
            if flag == 0:
                x[1] = 0
    print(x_white_list)

    # 最左边是省份代号，长度必定大于30，但“川”字需要特殊处理
    flag = 0
    for i in range(len(x_white_list)):
        x = x_white_list
        if x[i][1] < 30:
            if flag == 0:
                if x[i+1][1] < 30 and x[i+2][1] < 30 and x[i+2][0]+x[i+2][1]-x[i][0] < 55:
                    x_white_list[i][1] = x[i+2][0]+x[i+2][1]-x[i][0]
                    x_white_list[i+1][1] = 0
                    x_white_list[i+2][1] = 0
                    flag = 1
                else:
                    x_white_list[i][1] = 0
            else:
                x_white_list[i][1] = 0
        else:
            break
    
    x_white_list.sort(key=lambda x:x[1], reverse=True)
    x_char_list = x_white_list[:7]
    x_char_list.sort()
    print(x_char_list)
    

    # 将每个字符存入单独的图像中
    img_array = []
    for x_char in x_char_list:
        img_array.append(y_crop_img[:,x_char[0]:x_char[0]+x_char[1]])
    for i in range(len(img_array)):
        cv2.imshow(str(i), img_array[i])
        

    new_img_array = [makeImgSquare(x) for x in img_array]
    
    
    
    
    pil_array = [Image.fromarray(cv2.cvtColor(x,cv2.COLOR_GRAY2RGB)) for x in new_img_array]
    result = ''.join(identify(pil_array))
    return result

def makeImgSquare(img):
    height, width = img.shape[:2]
    square_length = height
    new_img = np.zeros((square_length, square_length, 1), np.uint8)
    for i in range(square_length):
        for j in range(width):
            col = j + int((square_length-width) / 2)
            new_img[i,col] = img[i,j]
    new_img = cv2.resize(new_img, (20,20), interpolation=cv2.INTER_LINEAR)
    return new_img

def recognition(path):
    im = cv2.imread(path)
    # im = cv2.imread('./imgs/pictures/42.jpg')
    height, width = im.shape[:2]
    plate_color_img = preprocess(im)
    # cv2.imshow('pci',plate_color_img)
    rect, plate = findPlate(plate_color_img, im)
    cv2.drawContours(im,[rect],-1,(0,255,0),3)
    cv2.imshow('im',im)
    cv2.imshow('plate', plate)



    plate_binary = cv2.cvtColor(plate,cv2.COLOR_BGR2GRAY)
    ret, plate_binary = cv2.threshold(plate_binary, 0, 255, cv2.THRESH_OTSU)
    cv2.imshow('binary', plate_binary)

    result = getChar(plate_binary)
    print(result)

    # cv2.waitKey()
    return result

if __name__ == "__main__":
    path = input('Please input path:')
    recognition(path)