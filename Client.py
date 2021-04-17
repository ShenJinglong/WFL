
import numpy as np
import params

from CommClient import CommClient
from TFTrainer import TFTrainer as TR

def compute_norm(data):
    return sum([np.sum(item**2) for item in data])

class Client():
    def __init__(self) -> None:
        pass

    def run(self, data, label):
        self.__comm = CommClient('127.0.0.1', 12345)
        self.__trainer = TR()
        self.__hi = np.random.uniform()
        self.__transmit_power = params.CLIENT_TRANSMIT_POWER
        
        for _ in range(params.ITERATION_NUM):
            # 接收来自服务器的 Global Model
            global_model = self.__comm.recv()
            # 计算梯度
            grad = self.__trainer.compute_gradient(global_model, data, label)
            # 计算梯度的二范数
            grad_norm = compute_norm(grad)
            # 向服务器发送结果
            self.__comm.send({'grad_norm': grad_norm, 'received_power': self.__hi * self.__transmit_power})
            # 接收服务器的调度结果：1为调度，0为未调度
            sche_sig = self.__comm.recv()
            if sche_sig == 1:
                # 被调度后更新模型，得到 local model
                self.__trainer.train_with_grad(grad)
                # 向服务器发送 local model
                self.__comm.send(self.__trainer.get_weights())

if __name__ == '__main__':
    client = Client()
    client.run()