
import numpy as np
import params

from CommClient import CommClient
from TFTrainer import TFTrainer as TR

def compute_norm(data):
    return sum([np.sum(item**2) for item in data])

"""
    @brief: 极坐标转欧氏坐标
    @param [polar_coordinate]: 要转换的极坐标 | 都是用普通列表表示的坐标
    @return: 转换结果（欧氏坐标）
"""
def polar2euclid(polar_coordinate):
    return [polar_coordinate[0] * np.math.cos(polar_coordinate[1]), polar_coordinate[0] * np.math.sin(polar_coordinate[1])]

"""
    @brief: 欧氏坐标转极坐标
    @param [polar_coordinate]: 要转换的欧氏坐标 | 都是用普通列表表示的坐标
    @return: 转换结果（极坐标）
"""
def euclid2polar(euclid_coordinate):
    return [np.math.sqrt(euclid_coordinate[0]**2 + euclid_coordinate[1]**2), np.math.atan2(euclid_coordinate[1], euclid_coordinate[0])]

class Client():
    def __init__(self) -> None:
        pass

    def run(self, data, label, p_d):
        self.__comm = CommClient('127.0.0.1', 12345)
        self.__trainer = TR()

        self.__polar_position = p_d[0]
        self.__polar_direction = p_d[1]
        self.__euclid_position = polar2euclid(self.__polar_position)
        self.__euclid_direction = polar2euclid(self.__polar_direction)

        self.__hi = self.__polar_position[0]**(-params.PATHLOSS_FACTOR)
        self.__transmit_power = params.CLIENT_TRANSMIT_POWER
        
        for _ in range(params.ITERATION_NUM):
            # 接收来自服务器的 Global Model
            global_model = self.__comm.recv()
            # 计算梯度
            grad = self.__trainer.compute_gradient(global_model, data, label)
            # 计算梯度的二范数
            grad_norm = compute_norm(grad)
            # 向服务器发送结果
            self.__comm.send({'grad_norm': grad_norm, 'received_power': self.__hi * self.__transmit_power, 'position': self.__euclid_position})
            # 接收服务器的调度结果：1为调度，0为未调度
            sche_sig = self.__comm.recv()
            if sche_sig == 1:
                # 被调度后更新模型，得到 local model
                self.__trainer.train_with_grad(grad)
                # 向服务器发送 local model
                self.__comm.send(self.__trainer.get_weights())
            self.__update_user()
            
    def __update_user(self):
        self.__move(1)
        self.__hi = self.__polar_position[0]**(-params.PATHLOSS_FACTOR)

    def __move(self, time_elapsed):
        distance = self.__polar_direction[0] * time_elapsed
        pose_d = polar2euclid([distance, self.__polar_direction[1]])
        self.__euclid_position[0] += pose_d[0]
        self.__euclid_position[1] += pose_d[1]

        self.__polar_position = euclid2polar(self.__euclid_position)

        if self.__polar_position[0] > 100:
            normal_dir = polar2euclid([1, self.__polar_position[1]])
            dot_product = self.__euclid_direction[0] * normal_dir[0] + self.__euclid_direction[1] * normal_dir[1]
            polar_rho_vec = [dot_product, self.__polar_position[1]]
            euclid_rho_vec = polar2euclid(polar_rho_vec)
            euclid_side_vec = [self.__euclid_direction[0] - euclid_rho_vec[0], self.__euclid_direction[1] - euclid_rho_vec[1]]
            self.__euclid_direction[0], self.__euclid_direction[1] = euclid_side_vec[0] - euclid_rho_vec[0], euclid_side_vec[1] - euclid_rho_vec[1]
            self.__polar_direction = euclid2polar(self.__euclid_direction)

if __name__ == '__main__':
    client = Client()
    client.run()