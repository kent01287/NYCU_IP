import numpy as np
import pandas as pd

import os
import cv2



"""
TODO Binary transfer
Simply to only foreground(255) and background(0)
"""
    
def to_binary(img):
    #Covert the image to gray style
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Apply the threshold to the image
    thersohold = 128
    binary = (gray<thersohold)*255
    binary = binary.astype(np.uint8)
    return binary

"""
TODO Two-pass algorithm
對於每個前景(value = 255) check neighbors pixels 
如果已有標籤 取最小像素
如果沒有標籤 則分配新的標籤
如果有多個標籤 則記錄等價標籤

第二次
使用第一階段的等價標籤 合併為統一標籤
再次掃描整個圖像 將所有像素替換合併後的標籤
"""


def two_pass(binary_img, connectivity=4):
    """
    
    Parameters:
        binary_img: Binary image (values 0 or 255).
        connectivity: Connectivity, either 4 or 8.
    
    Returns:
        Labeled image with unique labels.
    """
    if connectivity == 4:
        # Consider the left and top neighbors
        neighbors = [(-1, 0), (0, -1)]
    elif connectivity == 8:
        # Consider the left, top-left, top, and top-right neighbors
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1)]
    else:
        raise ValueError("connectivity必須為4或8")

    # Initialize the labeled image
    labeled_img = np.zeros_like(binary_img, dtype=int)
    
    # Record the equivalences between labels
    equivalences = []
    next_label = 1

    # first pass : scan from top-left to bottom-right
    rows, cols = binary_img.shape
    for r in range(rows):
        for c in range(cols):
            if binary_img[r, c] != 0:
                # for image label == 1
                neighbor_labels = []
                for dr, dc in neighbors:
                    # get the neighbor pixel
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < rows and 0 <= cc < cols:
                        if labeled_img[rr, cc] > 0:
                            neighbor_labels.append(labeled_img[rr, cc])

                if len(neighbor_labels) == 0:
                    # assgin new label
                    labeled_img[r, c] = next_label
                    next_label += 1
                else:
                    # use neighbor new label
                    min_label = min(neighbor_labels)
                    labeled_img[r, c] = min_label
                    # record the equivalences if there are multiple labels
                    for lb in neighbor_labels:
                        if lb != min_label:
                            equivalences.append((min_label, lb))
    
    label_sets = {}
    for label in range(1, next_label):
        label_sets[label] = {label}
    
    for (a, b) in equivalences:
        # 將a所在集合與b所在集合合併
        set_a = None
        set_b = None
        # find the label a 's set and label b 's set
        for s in label_sets.values():
            if a in s:
                set_a = s
            if b in s:
                set_b = s

        if set_a is not None and set_b is not None and set_a is not set_b:
            #union the set
            union_set = set_a.union(set_b)
            
            for lbl in union_set:
                label_sets[lbl] = union_set
    # Ex: {1:{1,2},2:{2,3}} -> {1:{1,2,3},2:{1,2,3}}

    label_map = {}
    for label in range(1, next_label):
        # Find the min as the representative Ex: {1,2,3} -> 1
        root = min(label_sets[label])
        label_map[label] = root
    
    for r in range(rows):
        for c in range(cols):
            if labeled_img[r, c] > 0:
                labeled_img[r, c] = label_map[labeled_img[r, c]]

    return labeled_img



"""
TODO Seed filling algorithm
1.initialization the label matrix to zero
2.從左上到右下scan image
3.如果是前景pixel 且未被標籤 設定新label 並設為seed 
4.檢查seed的neighbors 如果是前景且未被標籤 則設定為seed
5.return label matrix
seed 會越來越大
"""
def seed_filling(binary_img, connectivity=4):
    """
    Perform connected component labeling using Seed Filling algorithm.

    Parameters:
        binary_img (numpy.ndarray): Binary image (values 0 or 255).
        connectivity (int): Connectivity, either 4 or 8.

    Returns:
        numpy.ndarray: Label matrix with unique labels for each connected component.
    """
    H, W = binary_img.shape
    labels = np.zeros((H, W), dtype=np.int32)
    current_label = 1

    def is_valid(x, y):
        return 0 <= x < H and 0 <= y < W and binary_img[x, y] == 255 and labels[x, y] == 0

    for x in range(H):
        for y in range(W):
            if binary_img[x, y] == 255 and labels[x, y] == 0:  
                stack = [(x, y)]
                while stack:
                    cx, cy = stack.pop()
                    left, right = cy, cy
                    while left > 0 and is_valid(cx, left - 1):
                        left -= 1
                    while right < W - 1 and is_valid(cx, right + 1):
                        right += 1

                    for i in range(left, right + 1):
                        labels[cx, i] = current_label
                        if cx > 0 and is_valid(cx - 1, i):  
                            stack.append((cx - 1, i))
                        if cx < H - 1 and is_valid(cx + 1, i):  
                            stack.append((cx + 1, i))
                        if connectivity == 8:
                            if cx > 0 and i > 0 and is_valid(cx - 1, i - 1):
                                stack.append((cx - 1, i - 1))
                            if cx > 0 and i < W - 1 and is_valid(cx - 1, i + 1):
                                stack.append((cx - 1, i + 1))
                            if cx < H - 1 and i > 0 and is_valid(cx + 1, i - 1):
                                stack.append((cx + 1, i - 1))
                            if cx < H - 1 and i < W - 1 and is_valid(cx + 1, i + 1):
                                stack.append((cx + 1, i + 1))

                current_label += 1  

    return labels
    
    


"""
Bonus
"""
def other_cca_algorithm(binary_img,connectivity=4):
    '''implement flood fill'''
    H,W = binary_img.shape
    labels=np.zeros((H,W),dtype=np.int32)
    current_label=1
    
    def fill(x, y):
        stack = [(x, y)]
        while stack:
            x, y = stack.pop()
            if labels[y, x] == 0 and binary_img[y, x] == 255:  
                labels[y, x] = current_label

                if y > 0 and labels[y - 1, x] == 0: stack.append((x, y - 1))
                if y < H - 1 and labels[y + 1, x] == 0: stack.append((x, y + 1))
                if x > 0 and labels[y, x - 1] == 0: stack.append((x - 1, y))
                if x < W - 1 and labels[y, x + 1] == 0: stack.append((x + 1, y))
                
                if connectivity == 8: 
                    if y > 0 and x > 0 and labels[y - 1, x - 1] == 0: stack.append((x - 1, y - 1))
                    if y > 0 and x < W - 1 and labels[y - 1, x + 1] == 0: stack.append((x + 1, y - 1))
                    if y < H - 1 and x > 0 and labels[y + 1, x - 1] == 0: stack.append((x - 1, y + 1))
                    if y < H - 1 and x < W - 1 and labels[y + 1, x + 1] == 0: stack.append((x + 1, y + 1))
    
    for y in range(H):
        for x in range(W):
            if binary_img[y, x] == 255 and labels[y, x] == 0:
                fill(x, y)
                current_label += 1
    print(labels)
    return labels


"""
TODO Color mapping
"""
def color_mapping(labels):
    # Create an empty RGB image
    H, W = labels.shape
    color_img = np.zeros((H, W, 3), dtype=np.uint8)


    # Assign a random color to each label
    unique_labels = np.unique(labels) # Get all unique labels
    label_to_color = {label: tuple(np.random.randint(0, 255, size=3)) for label in unique_labels if label > 0} 
    # Ex : {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255)}

    for y in range(H):
        for x in range(W):
            if labels[y, x] > 0:
                color_img[y, x] = label_to_color[labels[y, x]]

    return color_img


"""
Main function
"""
def main():

    os.makedirs("result/connected_component/two_pass", exist_ok=True)
    os.makedirs("result/connected_component/seed_filling", exist_ok=True)
    os.makedirs("result/connected_component/flood_filling", exist_ok=True)
    connectivity_type = [4,8]
    np.random.seed(42) # make sure the color will be the same
    for i in range(2):
        img = cv2.imread("data/connected_component/input{}.png".format(i + 1))

        for connectivity in connectivity_type:

            # TODO Part1: Transfer to binary image
            binary_img = to_binary(img)

            # TODO Part2: CCA algorithm
            two_pass_label = two_pass(binary_img, connectivity)
            seed_filling_label = seed_filling(binary_img, connectivity)
            flood_filling_label = other_cca_algorithm(binary_img,connectivity)

            # TODO Part3: Color mapping       
            two_pass_color = color_mapping(two_pass_label)
            seed_filling_color = color_mapping(seed_filling_label)
            flood_filling_color = color_mapping(flood_filling_label)
            
            cv2.imwrite("result/connected_component/two_pass/input{}_c{}.png".format(i + 1, connectivity), two_pass_color)
            print(f"The two pass complete{i+1} and connectivity {connectivity}")
            cv2.imwrite("result/connected_component/seed_filling/input{}_c{}.png".format(i + 1, connectivity), seed_filling_color)
            print(f"The seed filling complete{i+1} and connectivity {connectivity}")
            cv2.imwrite("result/connected_component/flood_filling/input{}_c{}.png".format(i + 1, connectivity), flood_filling_color)

if __name__ == "__main__":
    main()