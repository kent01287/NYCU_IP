import cv2
import numpy as np

'''
    通常 去躁 去模糊都要轉成frequency domain去做 有以下優點
    1. 降低計算量 可轉成簡單的乘法 而不是convolution
    2. 高頻成分 代表細節 低頻 代表 大範圍結構
    3. 模糊圖像通常損失高頻成分 而清晰圖像有高頻成分
'''

def estimate_psf_angle_length(img_blurred):
    edges = cv2.Canny(img_blurred, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
    if lines is not None:
        angles = [line[0][1] for line in lines]
        angle = np.median(angles)  # 使用中位數避免極端值影響
        length = max([line[0][0] for line in lines])  # 取最大線段長度作為參考
        return np.degrees(angle), int(length)
    return None, None
    

def generate_motion_blur_psf(size=2,length=20,angle=45):
    '''
        描述導致圖像模糊的運動模糊點擴展函數 Point Spread Function (PSF)
        size : 15~45
        length : 10~30
        angle : 0~180
        psf 非0 模糊影響主要位置
    '''
    psf = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    for i in range (length):
        x = int(center + i * np.cos(np.radians(angle)))
        y = int(center + i * np.sin(np.radians(angle)))
        if 0<=x<size and 0<=y<size:
            psf[y, x] = 1.0 
    psf /= psf.sum() # normalization
    return psf
    
    



def wiener_filtering(img_blurred, psf, k=0.1):
    """
    Apply Wiener filtering to an RGB image.
    """
    # Initialize the restored image
    img_restored = np.zeros_like(img_blurred)
    
    # Get the spatial dimensions
    H, W = img_blurred.shape[:2]
    
    
    # Compute the FFT of the padded PSF
    psf_fft = np.fft.fft2(psf , s= (H,W))
    psf_fft_conj = np.conj(psf_fft)
    psf_power = np.abs(psf_fft) ** 2
    wiener_filter = psf_fft_conj / (psf_power + k)
    
    # Apply the Wiener filter to each channel
    for c in range(3):
        # Transfer to frequency domain
        img_blurred_fft = np.fft.fft2(img_blurred[:, :, c])
        
        # Apply the Wiener filter
        img_restored_fft = img_blurred_fft * wiener_filter
        
        # Convert back to spatial domain and take the absolute value
        img_restored_channel = np.abs(np.fft.ifft2(img_restored_fft))
        
        # Clip the values and convert to uint8
        img_restored[:, :, c] = np.clip(img_restored_channel, 0, 255).astype(np.uint8)
    
    return img_restored
    
    




def constrained_least_square_filtering(img_blurred, psf, reg_param=0.1):
    # Initialize the restored image
    img_restored = np.zeros_like(img_blurred)
    
    # Get the spatial dimensions
    H, W = img_blurred.shape[:2]
    
    
    # Compute the FFT of the padded PSF
    psf_fft = np.fft.fft2(psf , s= (H,W))
    psf_fft_conj = np.conj(psf_fft)
    psf_power = np.abs(psf_fft) ** 2
    laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    laplacian_fft = np.fft.fft2(laplacian, s=(H, W))
    cls_filiter = psf_fft_conj / (psf_power + reg_param * laplacian_fft )
    
    # Apply the Wiener filter to each channel
    for c in range(3):
        # Transfer to frequency domain
        img_blurred_fft = np.fft.fft2(img_blurred[:, :, c])
        
        # Apply the cls filter
        img_restored_fft = img_blurred_fft * cls_filiter 
        
        # Convert back to spatial domain and take the absolute value
        img_restored_channel = np.abs(np.fft.ifft2(img_restored_fft))
        
        # Clip the values and convert to uint8
        img_restored[:, :, c] = np.clip(img_restored_channel, 0, 255).astype(np.uint8)
    
    return img_restored


def inverse_filtering(img_blurred, psf):
   # Initialize the restored image
    img_restored = np.zeros_like(img_blurred)
    
    # Get the spatial dimensions
    H, W = img_blurred.shape[:2]
    
    
    # Compute the FFT of the padded PSF
    psf_fft = np.fft.fft2(psf , s= (H,W))
   
    inverse_filiter = psf_fft+1e-8
    
    # Apply the Wiener filter to each channel
    for c in range(3):
        # Transfer to frequency domain
        img_blurred_fft = np.fft.fft2(img_blurred[:, :, c])
        
        # Apply the inverse filter
        img_restored_fft = img_blurred_fft * inverse_filiter 
        
        # Convert back to spatial domain and take the absolute value
        img_restored_channel = np.abs(np.fft.ifft2(img_restored_fft))
        
        # Clip the values and convert to uint8
        img_restored[:, :, c] = np.clip(img_restored_channel, 0, 255).astype(np.uint8)
        
    return img_restored


def compute_PSNR(image_original, image_restored):
    #PSNR 越高越好
    # PSNR = 10 * log10(max_pixel^2 / MSE)
    psnr = 10 * np.log10(255 ** 2 / np.mean((image_original.astype(np.float64) - image_restored.astype(np.float64)) ** 2))

    return psnr

def main():
    for i in range(2):
        img_original = cv2.imread("data/image_restoration/testcase{}/input_original.png".format(i + 1))
        img_blurred = cv2.imread("data/image_restoration/testcase{}/input_blurred.png".format(i + 1))
        degree,length =  estimate_psf_angle_length(img_blurred)
        print(f"degree:{degree},length:{length}")
        psf = generate_motion_blur_psf(length=length,angle=degree)
        wiener_img = wiener_filtering(img_blurred, psf, k=0.01)
        #cv2.imwrite(f"./result/Wiener_{i}.jpg",np.vstack([img_original,wiener_img]))
        constrained_least_square_img = constrained_least_square_filtering(img_blurred, psf, reg_param=0.01)
        #cv2.imwrite(f"./result/Cls_{i}.jpg",np.vstack([img_original,constrained_least_square_img]))
        inverse_filtering_img = inverse_filtering(img_blurred, psf)
        #cv2.imwrite(f"./result/Inverse_{i}.jpg",np.vstack([img_original,inverse_filtering_img]))

        print("\n---------- Testcase {} ----------".format(i))
        print("Method: Wiener filtering")
        print("PSNR = {}\n".format(compute_PSNR(cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY), cv2.cvtColor(wiener_img, cv2.COLOR_BGR2GRAY))))
        print("Method: Constrained least squares filtering")
        print("PSNR = {}\n".format(compute_PSNR(cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY), cv2.cvtColor(constrained_least_square_img, cv2.COLOR_BGR2GRAY))))
        print("Method: Inverse filtering")
        print("PSNR = {}\n".format(compute_PSNR(cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY), cv2.cvtColor(inverse_filtering_img, cv2.COLOR_BGR2GRAY))))
        cv2.imshow("Restoration Results", np.hstack([img_blurred, wiener_img,constrained_least_square_img,inverse_filtering_img]))
        cv2.waitKey(0)

if __name__ == "__main__":
    main()


