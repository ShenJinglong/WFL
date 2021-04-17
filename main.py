import multiprocessing as mp
import numpy as np

import params

from Client import Client
from Server import Server
from MNIST.MNISTReader import MNISTImageReader as ds_r
from MNIST.MNISTReader import MNISTLabelReader as lb_r

if __name__ == '__main__':
    server = Server()
    clients = []
    for _ in range(params.CLIENT_NUM):
        clients.append(Client())

    dr = ds_r(params.MNIST_TRAIN_IMAGES_PATH)
    lr = lb_r(params.MNIST_TRAIN_LABELS_PATH)
    dr.open()
    lr.open()
    index, data = dr.read(params.DATASET_SIZE_USED_TO_TRAIN)
    data = data[..., np.newaxis] / 255.0
    index, label = lr.read(params.DATASET_SIZE_USED_TO_TRAIN)
    dr.close()
    lr.close()

    edr = ds_r(params.MNIST_EVAL_IMAGES_PATH)
    elr = lb_r(params.MNIST_EVAL_LABELS_PATH)
    edr.open()
    elr.open()
    index, edata = edr.read(params.DATASET_SIZE_USED_TO_EVAL)
    edata = edata[..., np.newaxis] / 255.0
    index, elabel = elr.read(params.DATASET_SIZE_USED_TO_EVAL)
    edr.close()
    elr.close()

    server_p = mp.Process(target=server.run, args=(edata, elabel))
    clients_p = []
    local_data_size = int(np.floor(params.DATASET_SIZE_USED_TO_TRAIN / params.CLIENT_NUM))
    for i, client in enumerate(clients):
        clients_p.append(mp.Process(target=client.run, args=(
            data[i*local_data_size:(i+1)*local_data_size, :, :, :], 
            label[i*local_data_size:(i+1)*local_data_size]
        )))
    
    server_p.start()
    for client_p in clients_p:
        client_p.start()

    server_p.join()
    for client_p in clients_p:
        client_p.join()


    