import cv2
import os
import numpy as np


"""
TODO White patch algorithm
對顏色進行調整來表達自然性和正確性
white balance correction 確保影像中白色看起來是真的白色 而不帶有冷色調或暖色調
    有兩種方法 white patch algo : 最大pixel全部最大化成白色
    
Goal : 變亮 矯正光源條件不足
"""
def white_patch_algorithm(img):
    # get the maximum value of each channel
    max_R = np.max(img[:, :, 2])
    max_G = np.max(img[:, :, 1])
    max_B = np.max(img[:, :, 0])
    
    # 按比例放大
    img = img.astype(np.float32)
    img[:, :, 2] = img[:, :, 2] * (255 / max_R)
    img[:, :, 1] = img[:, :, 1] * (255 / max_G)
    img[:, :, 0] = img[:, :, 0] * (255 / max_B)
    
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


"""
TODO Gray-world algorithm
假設平均顏色是灰色 調整RGB的比例以達到均衡

Goal : 變暗 整體色調均衡 適合過曝的照片
"""
def gray_world_algorithm(img):
    # get the average value of each channel
    avg_R = np.mean(img[:, :, 2])
    avg_G = np.mean(img[:, :, 1])
    avg_B = np.mean(img[:, :, 0])
    
    # calculate the ratio of each channel
    mean_all = (avg_R+avg_G+avg_B)/3
    sacle_R = mean_all/avg_R
    sacle_G = mean_all/avg_G
    sacle_B = mean_all/avg_B
    
    # According to the ratio, adjust the value of each channel
    img = img.astype(np.float32)
    img[:, :, 2] = img[:, :, 2] * sacle_R
    img[:, :, 1] = img[:, :, 1] * sacle_G
    img[:, :, 0] = img[:, :, 0] * sacle_B
    
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)
    


"""
Bonus 
"""
def other_white_balance_algorithm(img): 
    """
    Gaussian-Based White Balance Algorithm.
    """
    # turn the image into float32
    img = img.astype(np.float32)


    for i in range(3):
        channel = img[:, :, i]

        # calculate the mean and standard deviation of the channel
        mean = np.mean(channel)
        std = np.std(channel)

        
        if std > 0:  
            img[:, :, i] = (channel - mean) / std * 64 + 128  # Remapping
        else:
            img[:, :, i] = channel  # Noting change

    # clip and convert the image back to uint8
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


"""
Main function
"""
def main():

    os.makedirs("result/color_correction", exist_ok=True)
    for i in range(2):
        img = cv2.imread("data/color_correction/input{}.bmp".format(i + 1))

        # TODO White-balance algorithm
        white_patch_img = white_patch_algorithm(img)
        gray_world_img = gray_world_algorithm(img)
        gaussian_img = other_white_balance_algorithm(img)

        cv2.imwrite("result/color_correction/white_patch_input{}.bmp".format(i + 1), white_patch_img)
        cv2.imwrite("result/color_correction/gray_world_input{}.bmp".format(i + 1), gray_world_img)
        cv2.imwrite("result/color_correction/gaussian_input{}.bmp".format(i + 1), gaussian_img)

if __name__ == "__main__":
    main()