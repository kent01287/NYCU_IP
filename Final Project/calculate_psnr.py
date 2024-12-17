import os
import numpy as np
import skimage.io as sio
from skimage.metrics import peak_signal_noise_ratio as psnr

def calculate_psnr(folder1, folder2):
    images1 = sorted(os.listdir(folder1))
    images2 = sorted(os.listdir(folder2))

    total_psnr = 0.0
    num_images = min(len(images1), len(images2))  # 確保兩個資料夾的圖片數量一致

    for i in range(num_images):
        image1_path = os.path.join(folder1, images1[i])
        image2_path = os.path.join(folder2, images2[i])

        img1 = sio.imread(image1_path)
        img2 = sio.imread(image2_path)

        # 確保圖片是 3 通道 (RGB)，如果是 4 通道 (RGBA)，則去除 Alpha 通道
        if img1.shape[-1] == 4:
            img1 = img1[:, :, :3]
        if img2.shape[-1] == 4:
            img2 = img2[:, :, :3]

        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

        psnr_value = psnr(img1, img2, data_range=1.0)
        total_psnr += psnr_value
        print(f"Image {images1[i]} PSNR: {psnr_value:.4f}")

    average_psnr = total_psnr / num_images
    print(f"\nAverage PSNR: {average_psnr:.4f}")

if __name__ == '__main__':
    folder1 = './Mcmaster18_Dataset/val_gt'  # Ground truth images
    folder2 = './Mcmaster18_Dataset/val_improved_output/'  # Generated images

    calculate_psnr(folder1, folder2)

# SIDD_images
# val_origin_output ：35.4359
# val_improved_output ：35.6511

# KodaK24_images
# val_origin_output ：26.9150
# val_improved_output ：21.1486

# Mcmaster18_images
# val_origin_output ：26.1827
# val_improved_output ：21.5663