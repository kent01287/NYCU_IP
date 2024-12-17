import numpy as np
import cv2
import argparse

'''
 三種常見的影像處理增強 
 1. Gaussian filter : 平滑圖像 主要去除噪聲 使用 5*5 matrix value 符合 高斯分佈 適合去除高斯噪聲
 2. Median filter: 每個 pixel 值取中位數 用於去除 salt-and-pepper noise
 3. Laplacian sharpening: 用於增強圖像的邊緣 透過對原始圖像進行高通濾波器操作 並將結果加回原始圖像 使用laplacian operator 3*3matrix 對中心附近的像素進行加權平均 並突出變化比較大的地方
 
'''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gaussian', action='store_true')
    parser.add_argument('--median', action='store_true')
    parser.add_argument('--laplacian', action='store_true')
    args = parser.parse_args()
    return args

def padding(input_img, kernel_size):
    ############### YOUR CODE STARTS HERE ###############
    """Applies zero padding to the input image."""
    '''
        選擇使用 zero padding 進行圖像處理
        np.pad mode='constant' 使用常數 edge: 使用邊緣pixel 來填充 reflect:進行鏡像的反射
        
    '''
    pad_size = kernel_size // 2
    output_img = np.pad(input_img, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='constant', constant_values=0)
    
    ############### YOUR CODE ENDS HERE #################
    return output_img

def convolution(input_img, kernel):
    ############### YOUR CODE STARTS HERE ###############
    """Applies a convolution operation using the given kernel."""
    kernel_size = kernel.shape[0]
    img_h,img_w,img_c = input_img.shape
    padded_img = padding(input_img, kernel_size)
    output_img = np.zeros_like(input_img)
    
    for i in range(img_h):
        for j in range(img_w):
            for c in range(img_c):  # Iterate over the color channels
                region = padded_img[i:i+kernel_size, j:j+kernel_size, c]
                output_img[i, j, c] = np.sum(region * kernel) # element-wise multiplication and summation
    
    ############### YOUR CODE ENDS HERE #################
    return output_img
def gaussian_kernel(size, sigma):
    """Generates a Gaussian kernel."""
    #高斯kernel 給定size 和 sigma 生成的kernel是唯一的
    kernel_1d = np.linspace(-(size // 2), size // 2, size) #生成一維數組 asuume size=5 -> [-2, -1, 0, 1, 2]
    gaussian = np.exp(-0.5 * (kernel_1d ** 2) / sigma ** 2) # 套用高斯函式的function
    kernel_1d = gaussian / gaussian.sum() # normalization
    kernel_2d = np.outer(kernel_1d, kernel_1d) # 透過 外積來生成二維高斯函數
    return kernel_2d / kernel_2d.sum() # normalization

def gaussian_filter(input_img):
    ############### YOUR CODE STARTS HERE ###############
    """Applies a Gaussian filter to the input image."""
    sigma = 0.5  # 0.5 2 5
    size = 3  # 3 5 7
    kernel = gaussian_kernel(size, sigma)
    ############### YOUR CODE ENDS HERE #################
    return convolution(input_img, kernel)

def median_filter(input_img):
    ############### YOUR CODE STARTS HERE ###############
    """Applies a Median filter to the input image."""
    '''
        1. 將圖像進行 zero padding
        2. 9個pixel 排列 取中位數
        3. 中間改成中位數的value
        
        優點 : 去躁效果好 保留邊緣細節不錯
        缺點 : 計算量大 對高斯噪聲效果不好
    '''
    size = 3  # kernel_size
    padded_img = padding(input_img, size)
    output_img = np.zeros_like(input_img)
    
    for i in range(input_img.shape[0]):
        for j in range(input_img.shape[1]):
            for c in range(input_img.shape[2]):  
                region = padded_img[i:i+size, j:j+size, c]  # Get the region of K*K size
                output_img[i, j, c] = np.median(region)
    ############### YOUR CODE ENDS HERE #################
    return output_img

def laplacian_sharpening(input_img):
    ############### YOUR CODE STARTS HERE ###############
    """Applies Laplacian sharpening to the input image."""
    '''
        增強圖像的邊緣細節，使圖像看起來更銳利，用於增強圖像的邊緣
        缺點 : 不適合有大量噪聲的圖片 銳化過強可能放大噪聲
    '''
    
    # 8-connected Laplacian kernel
    kernel = np.array([[-1, -1, -1],
                        [-1,  9, -1],
                        [-1, -1, -1]])

    # 4-connected Laplacian kernel
    kernel = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]])
    
    ############### YOUR CODE ENDS HERE #################
    return convolution(input_img, kernel)

if __name__ == "__main__":
    args = parse_args()

    if args.gaussian:
        input_img = cv2.imread("input_part1.jpg")
        output_img = gaussian_filter(input_img)
    elif args.median:
        input_img = cv2.imread("input_part1.jpg")
        output_img = median_filter(input_img)
    elif args.laplacian:
        input_img = cv2.imread("input_part2.jpg")
        output_img = laplacian_sharpening(input_img)

    cv2.imwrite("output.jpg", output_img)

    