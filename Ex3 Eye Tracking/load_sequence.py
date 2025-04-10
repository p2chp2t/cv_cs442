import matplotlib.pyplot as plt
import os

def load_sequence(sequence_name="sequence1"):
    seq_folder = sequence_name
    gt_folder = sequence_name + "_gt"

    image_files = sorted([f for f in os.listdir(seq_folder) 
                          if f.lower().endswith(('.png', '.jpg'))])    
    imgs = []
    seg_gts = []
    for fname in image_files:
        img_path = os.path.join(seq_folder, fname)
        img = plt.imread(img_path)
        imgs.append(img)
        gt_path = os.path.join(gt_folder, fname)
        if os.path.exists(gt_path):
            gt_img = plt.imread(gt_path)
            seg_gts.append(gt_img)
        else:
            seg_gts.append(None)
    
    blink_file_path = os.path.join(gt_folder, "gt_blink.txt") 
    blink_frames = []
    with open(blink_file_path, "r") as f:
        blink_frames = [int(x) for x in f.read().split()]
    
    return imgs, seg_gts, blink_frames

def load_edges_sequence(sequence_name = "sequence_canny"):
    seq_folder = sequence_name

    image_files = sorted([f for f in os.listdir(seq_folder) 
                          if f.lower().endswith(('.png', '.jpg'))])    
    imgs = []
    for fname in image_files:
        img_path = os.path.join(seq_folder, fname)
        img = plt.imread(img_path)
        imgs.append(img)
    return imgs

def visualize_sequence(imgs, rows=4):
    nb_img_per_row = len(imgs) // rows
    imgs = imgs[:rows * nb_img_per_row]
    fig, axs = plt.subplots(rows, nb_img_per_row, figsize=(20, 5))
    for i in range(len(imgs)):
        axs[i // nb_img_per_row, i % nb_img_per_row].imshow(imgs[i])
        axs[i // nb_img_per_row, i % nb_img_per_row].axis('off')
        axs[i // nb_img_per_row, i % nb_img_per_row].set_title(f"Image {i+1}")
    plt.tight_layout()
    plt.suptitle("Image Sequence")
    plt.show()