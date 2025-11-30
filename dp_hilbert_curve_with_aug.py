import time 
import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve
import tensorflow as tf
from tqdm import tqdm
import os

# Step 1: Normalize to [0, 1]
# Step 2: 
# Step 3: Pad to 32×32 (using 0, since Fashion-MNIST background is black)
# Step 4: Standardize (subtract mean, divide std)

print(f"--- SCRIPT EXECUTION STARTED AT: {time.time()} ---")

def process_2d_cnn_data(x_train_norm, x_test_norm, output_dir):
    
    mean_2d = np.mean(x_train_norm)
    std_2d = np.std(x_train_norm)
    print(f"2D (28x28) Mean: {mean_2d:.4f}, Std: {std_2d:.4f}")
    
    x_train_standardized = (x_train_norm - mean_2d) / std_2d
    x_test_standardized = (x_test_norm - mean_2d) / std_2d
    
    X_train_2d_np = x_train_standardized[..., np.newaxis]
    X_test_2d_np = x_test_standardized[..., np.newaxis]
    
    print(f"Final 2D training set shape: {X_train_2d_np.shape}")  # (60000, 28, 28, 1)
    print(f"Final 2D testing set shape: {X_test_2d_np.shape}")    # (10000, 28, 28, 1)
    
    np.save(os.path.join(output_dir, 'X_train_2d_28.npy'), X_train_2d_np)
    np.save(os.path.join(output_dir, 'X_test_2d_28.npy'), X_test_2d_np)
    print("2D CNN .npy files saved.")


# place image in the center, pad it to 32*32 to use hilbert curve
def pad_center(image, target_shape = (32, 32), pad_value = 0):
    pad_h = target_shape[0] - image.shape[0]
    pad_w = target_shape[1] - image.shape[1]
    pad_top, pad_left = pad_h // 2, pad_w // 2
    pad_down, pad_right = pad_h - pad_top, pad_w - pad_left

    return np.pad(image, ((pad_top, pad_down), (pad_left, pad_right)), mode = 'constant', constant_values= pad_value )



# using hilbert curve, #(2^5 = 32)
def convert_to_hilbert_1d(image_32):
    p = 5
    n = 2  
    hilbert_curve = HilbertCurve(p, n)
    sequence_1d = np.zeros(32*32, dtype= image_32.dtype)

    for i in range(32):
        for j in range(32):
            hilbert_index = hilbert_curve.distance_from_point([i,j])
            sequence_1d[hilbert_index] = image_32[i,j]

    return sequence_1d

def process_1d_tsc_data(x_train_norm, x_test_norm, y_train, y_test, output_dir):
    print("Padding to 32*32")
    padded_images_train = []
    for i in tqdm(x_train_norm, desc = 'Padding Train set'):
        padded_img_train = pad_center(i)
        padded_images_train.append(padded_img_train)
    x_train_padded = np.array(padded_images_train)


    padded_images_test = []
    for i in tqdm(x_test_norm, desc = 'Padding test set'):
        padded_img_test = pad_center(i)
        padded_images_test.append(padded_img_test)
    x_test_padded = np.array(padded_images_test)

    print("Standardization")
    mean_1d = np.mean(x_train_padded)
    std_1d = np.std(x_train_padded)
    print(f"Mean: {mean_1d:.4f}, Std: {std_1d:.4f}")

    x_train_standardized = (x_train_padded - mean_1d) / std_1d
    x_test_standardized = (x_test_padded - mean_1d) / std_1d

    print("Convert 2D images to 1D sequences using hilbert curve")
    x_train_1d = np.array([convert_to_hilbert_1d(img) for img in tqdm(x_train_standardized, desc="Converting training set to 1D")])
    x_test_1d = np.array([convert_to_hilbert_1d(img) for img in tqdm(x_test_standardized, desc= "Converting test set to 1d")])

    X_train_1d_np = x_train_1d[:, np.newaxis, :]
    X_test_1d_np = x_test_1d[:, np.newaxis, :]
    # X_train_padded_2d = x_train_padded[:, np.newaxis, :, :]
    # X_test_padded_2d = x_test_padded[:, np.newaxis, :, :]
    Y_train_np = y_train.astype(np.int64)
    Y_test_np = y_test.astype(np.int64)

    # check shape
    print(f"Final training set shape-1d: {X_train_1d_np.shape}")
    print(f"Final testing set shape-1d: {X_test_1d_np.shape}")
    # print(f"Final training set shape-2d: {X_train_padded_2d.shape}")
    # print(f"Final testing set shape-2d: {X_test_padded_2d.shape}")
    print(f"Final training set label: {Y_train_np.shape}")
    print(f"Final testing set label: {Y_test_np.shape}")

    print("Saving files...")
   
    np.save(os.path.join(output_dir, 'X_train_1d_aug.npy'), X_train_1d_np)
    np.save(os.path.join(output_dir, 'X_test_1d_aug.npy'), X_test_1d_np)
    # np.save(os.path.join(output_dir, 'X_train_2d_32.npy'), X_train_padded_2d)
    # np.save(os.path.join(output_dir, 'X_test_2d_32.npy'), X_test_padded_2d)
    np.save(os.path.join(output_dir, 'Y_train.npy'), Y_train_np)
    np.save(os.path.join(output_dir, 'Y_test.npy'), Y_test_np)
    print("np formate of x_train,x_test,y_train,y_test are done, both 1d and 2d ")

    # train_lite = np.hstack((Y_train_np[:, None], x_train_1d))
    # test_lite = np.hstack((Y_test_np[:, None], x_test_1d))
    # np.savetxt(os.path.join(output_dir, 'fashion_mnist_train.tsv'), train_lite, delimiter= '\t', fmt='%f')
    # np.savetxt(os.path.join(output_dir, 'fashion_mnist_test.tsv'), test_lite, delimiter= '\t', fmt='%f')

    print(f"1d Data processing is done, Files can be found in {os.path.abspath(output_dir)}")

def process_dataset():
    print("Start data processing")

    print("Load Fashion-Minist Dataset")
    (x_train, y_train),(x_test,y_test) = tf.keras.datasets.fashion_mnist.load_data()
    print("x_train_shape", x_train.shape)
    print("y_train_shape", y_train.shape)
    print("x_test_shape", x_test.shape)
    print("y_test_shape", y_test.shape)

    # [0,1]
    x_train_norm = x_train.astype('float32')/255.0
    x_test_norm = x_test.astype('float32')/255.0

    mean_base = np.mean(x_train_norm)
    std_base  = np.std(x_train_norm)
    print(f"Base Mean={mean_base:.4f}, Std={std_base:.4f}")

    print("Data Augmentation for training set")
    data_aug = tf.keras.Sequential(
        [tf.keras.layers.Input(shape=(28,28,1)),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),    # ±10° approx
        tf.keras.layers.RandomTranslation(0.05, 0.05)
        ]
    )

    x_train_aug = data_aug(x_train_norm[..., np.newaxis])
    x_train_aug = np.squeeze(x_train_aug, axis=-1)

    x_train_aug = np.clip(x_train_aug, 0.0, 1.0)


    output_dir = './processed_data_aug'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # process_2d_cnn_data(x_train_norm, x_test_norm, output_dir)
    process_1d_tsc_data(x_train_aug, x_test_norm, y_train, y_test, output_dir)


if __name__ == '__main__':
    process_dataset()
    


