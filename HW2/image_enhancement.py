import cv2
import numpy as np


"""
TODO Part 1: Gamma correction
非線性的圖像增強技術 ， 用來調整圖像的亮度
 output = ((input/255)^1/gamma)* 255
 gamma 越大 像素值提升到一個次方 亮度值的增長變緩慢 所以變更暗
"""
def gamma_correction(img, gamma=1.0):
    # normalize to [0~1]
    img_normalized = img.astype(np.float32) / 255.0
    # apply gamma correction
    img_gamma_corrected = np.power(img_normalized, 1.0 / gamma)
    # remapping
    img_gamma_corrected = img_gamma_corrected * 255.0

    img_gamma_corrected = img_gamma_corrected.astype(np.uint8)
    return img_gamma_corrected 


"""
TODO Part 2: Histogram equalization
增強圖像對比的技術，特別用來亮度不均勻的圖像
直方圖 表示每個灰度值的像素數量 當某一亮度範圍內的像素數量過多或過少時，圖像對比度會降低

GOAL 亮度均勻化

YCrCb color space : Y channel (luminance)亮度 Cr channel (red difference) Cb channel (blue difference) 
"""
def histogram_equalization(img):
    # Convert image to YCrCb color space
    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb) #維度不會變 [H,W,3] -> [H,W,3] RGB->Y Cr Cb
    # Equalize the histogram of the Y channel
    ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0]) # 對Y channel進行直方圖均衡化 重新分配亮度
    # Convert back to BGR color space
    return cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)


"""
Bonus :

"""
def other_enhancement_algorithm(img, sigma=1.0, strength=10):
    """
    Unsharp Masking 實現
    :param img: 原始圖像
    :param sigma: 高斯模糊的標準差，用於控制模糊程度
    :param strength: 銳化強度，值越大，邊緣增強效果越強
    :return: 銳化後的圖像
    """
    # apply Gaussian blur to the image
    blurred_img = cv2.GaussianBlur(img, (0, 0), sigma)
    
    # calculate the enhanced image 
    enhanced_img = cv2.addWeighted(img, 1 + strength, blurred_img, -strength, 0)
    
    return enhanced_img


"""
Main function
"""
def main():
    img = cv2.imread("data/image_enhancement/input.bmp")

    # TODO: modify the hyperparameter
    gamma_list = [0.5, 1, 1.5] # gamma value for gamma correction

    # TODO Part 1: Gamma correction
    for gamma in gamma_list:
        gamma_correction_img = gamma_correction(img)

        stacked_img = np.vstack([img, gamma_correction_img])
        cv2.imshow("Gamma correction | Gamma = {}".format(gamma), np.vstack([img, gamma_correction_img]))
        #cv2.imwrite("./result/Gamma_correction_Gamma_{}.jpg".format(gamma), np.vstack([img, gamma_correction_img]))
        cv2.waitKey(0)

        
    
    # TODO Part 2: Image enhancement using the better balanced image as input
    histogram_equalization_img = histogram_equalization(img)

    
    cv2.imshow("Histogram equalization", np.vstack([img, histogram_equalization_img]))
   # cv2.imwrite("./result/Histogram equalization.jpg", np.vstack([img, histogram_equalization_img]))
    cv2.waitKey(0)
    
    enhanced_img = other_enhancement_algorithm(img)
    cv2.imshow("Other enhancement method", np.vstack([img, enhanced_img]))
    #cv2.imwrite("./result/Other enhancement method.jpg", np.vstack([img, enhanced_img]))
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
