import os
import numpy as np
import torch
import gzip
import shutil
from tfrecord_lite import tf_record_iterator
from torch.utils.data import Dataset, DataLoader


def load_ds_from_dir(path):
    tensor_x = []
    tensor_y = []

    files = sorted(os.listdir(path))

    # lightning seems to work even if chunks are not equal size!
    for name_file in files:
        path_file = os.path.join(path, name_file)
        it = tf_record_iterator(path_file)
        data = next(it)
        x = np.frombuffer(data["x"][0], dtype='>u2')
        x = x.reshape(128, 128, 128, -1)
        x = np.rollaxis(x, 3, 0)
        x = x.astype(np.float32) / 255 - 0.5
        y = data["y"].astype(np.float32)
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        tensor_x.append(x)
        tensor_y.append(y)
        # print("in shape:", x.shape)
        # print("in max:", x.max())
        # print("y", y)

    tensor_x = torch.stack(tensor_x)

    # print(f"size dataset = {np.prod(tensor_x.shape) * 4 / (1024**2)}M")
    tensor_y = torch.stack(tensor_y)
    
    return tensor_x, tensor_y

def load_ds_from_dir_single(path):
    tensor_x = []
    tensor_y = []
    data_file = [name for name in os.listdir(path) if name.endswith("data.npy")]
    for name_file in data_file:
        path_file = os.path.join(path, name_file)
        x = np.load(path_file, allow_pickle=True)
        x = x.reshape(128, 128, 128, -1)
        x = np.rollaxis(x, 3, 0)
        x = x.astype(np.float32) / 500
        y = np.load(path_file.replace("data.npy", "label.npy"), allow_pickle=True)
        y = y.astype(np.float32) / 2 + 0.5
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        tensor_x.append(x)
        tensor_y.append(y)

    return tensor_x, tensor_y


class Load_Dataset_Cosmoflow(Dataset):
    def __init__(self, path, pre_load=False):
        self.path = path
        self.files = sorted(os.listdir(path))
        self.pre_load = pre_load
        if pre_load:
            self.x, self.y = load_ds_from_dir(path)
            

    def __getitem__(self, index):
        if self.pre_load:
            return self.x[index], self.y[index]
        else:
            name_file = self.files[index]
            path_file = os.path.join(self.path, name_file)
            it = tf_record_iterator(path_file)
            data = next(it)
            x = np.frombuffer(data["x"][0], dtype='>u2')
            x = x.reshape(128, 128, 128, -1)
            x = np.rollaxis(x, 3, 0)
            x = x.astype(np.float32) / 255 - 0.5
            y = data["y"].astype(np.float32)
            x = torch.from_numpy(x)
            y = torch.from_numpy(y)

            return x, y

    def __len__(self):
        return len(self.files)


def get_loader(batch_size=1, val_only=False, data_only=False, rank=0):
    train_set = Load_Dataset_Cosmoflow("/nfs/stak/users/gaosho/hpc-share/dataset/loxia/cosmo_decompressed/train", pre_load=True)
    test_set = Load_Dataset_Cosmoflow("/nfs/stak/users/gaosho/hpc-share/dataset/loxia/cosmo_decompressed/validation", pre_load=True) if rank == 0 else None
    
    if data_only:
        return train_set, test_set

    train_loader = None
    if not val_only:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    val_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader


def decompress_tfrecord(input_path, output_path):
    with gzip.open(input_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def decompress_folder_tfrecords(input_folder, output_folder):
    """Decompress all gzipped tfrecord files in a folder"""
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Iterate through all files in input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.tfrecord'):
            input_path = os.path.join(input_folder, filename)
            # Remove .gz extension for output file
            output_filename = filename # removes '.gz'
            output_path = os.path.join(output_folder, output_filename)
            
            print(f"Decompressing {filename}...")
            decompress_tfrecord(input_path, output_path)


if __name__ == "__main__":
    # Usage example
    tfrecord_dir = "/nfs/stak/users/gaosho/hpc-share/dataset/loxia/cosmoUniverse_2019_05_4parE_tf_v2_mini/validation"
    output_dir = "/nfs/stak/users/gaosho/hpc-share/dataset/loxia/cosmo_decompressed/validation"  # path to save decompressed files
    decompress_folder_tfrecords(tfrecord_dir, output_dir)
