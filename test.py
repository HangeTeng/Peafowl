import h5py

# 指定要创建的HDF5文件名
file_name = "large_dataset.hdf5"

# 创建一个HDF5文件，指定'w'模式以进行写入
with h5py.File(file_name, 'w') as file:
    # 这个文件在with语句块内创建并打开，之后会自动关闭。

    # 创建一个大的数据集，例如一个包含随机数据的数据集
    dataset_size = (10000, 100000)  # 指定数据集的大小
    dataset = file.create_dataset("large_data", shape=dataset_size, dtype='int64')

    # 将数据写入数据集，这里使用随机数据作为示例
    import numpy as np
    random_data = np.random.rand(*dataset_size).astype('float32')
    dataset[:] = random_data
