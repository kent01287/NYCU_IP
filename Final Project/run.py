import os

def run_inference():
    for i in range(10):
        image_name = f"val_noisy_{i}.png"
        command = (
            f"python infer.py --gpu 0 "
            f"--modelpath ./ckpts/temp5/CBSNlast.ckpt "
            f"--imagepath ./Mcmaster18_Dataset/val_noisy/{image_name} "
            f"--savepath ./Mcmaster18_Dataset/val_improved_output/"
        )
        print(f"Running inference on {image_name}...")
        os.system(command)
        print(f"Finished processing {image_name}\n")

if __name__ == '__main__':
    run_inference()
