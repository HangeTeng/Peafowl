from src.utils.h5dataset import HDF5Dataset
from src.utils.encoder import FixedPointEncoder

if __name__ == "__main__":
    # dataset
    examples = 6
    features = 60
    chunk = 100
    # sub_dataset
    nodes = 3
    sub_examples = examples * 5 // 6
    sub_features = features // nodes
    targets_rank = 0
    folder_path = "./data/SVM_{}_{}".format(
                examples, features)
    
    encoder = FixedPointEncoder()
    
    tgt = []
    tgt_folder_path = folder_path + "/tgt"
    for i in range(nodes):
        tgt_path = "{}/SVM_{}_{}_{}-{}_tgt.hdf5".format(tgt_folder_path,
                examples, features, i, nodes)
        tgt.append(HDF5Dataset(tgt_path))
    
    a = tgt[0].data[:,0]
    for i in range(1,nodes):
        a += tgt[i].data[:,0]
    print(encoder.decode(a))
