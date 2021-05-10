import numpy as np
import matplotlib.pyplot as plt

color1 = '#4b778d'
color2 = '#28b5b5'
color3 = '#8fd9a8'
color4 = '#d2e69c'

if __name__ == '__main__':
    random = np.load('./output/random.npz')
    algorithm1 = np.load('./output/algorithm1.npz')
    algorithm2 = np.load('./output/algorithm2.npz')

    font = {
        'size': 10
    }

    # x = np.mean(random['uplink_delay_recorder'])
    # y = np.mean(algorithm1['uplink_delay_recorder'])
    # z = np.mean(algorithm2['uplink_delay_recorder'])
    # print(x, y, z)
    # print((x - y) / x)
    # print((x - z) / x)
    # print((y - z) / y)

    # f, axes = plt.subplots(2, 2)
    # axes[0, 0].plot(random['iter_recorder'], random['uplink_delay_cost'], label='random')
    # axes[0, 0].plot(algorithm1['iter_recorder'], algorithm1['uplink_delay_cost'], label='algorithm1')
    # axes[0, 0].plot(algorithm2['iter_recorder'], algorithm2['uplink_delay_cost'], label='algorithm2')
    # axes[0, 0].legend()


    # axes[0, 1].plot(random['iter_recorder'], random['acc_recorder'], label='random')
    # axes[0, 1].plot(algorithm1['iter_recorder'], algorithm1['acc_recorder'], label='algorithm1')
    # axes[0, 1].plot(algorithm2['iter_recorder'], algorithm2['acc_recorder'], label='algorithm2')
    # axes[0, 1].legend()
    
    # axes[1, 0].plot(random['iter_recorder'], random['loss_recorder'], label='random')
    # axes[1, 0].plot(algorithm1['iter_recorder'], algorithm1['loss_recorder'], label='algorithm1')
    # axes[1, 0].plot(algorithm2['iter_recorder'], algorithm2['loss_recorder'], label='algorithm2')
    # axes[1, 0].legend()
    
    # axes[1, 1].plot(random['iter_recorder'], random['uplink_delay_recorder'], label='random')
    # axes[1, 1].plot(algorithm1['iter_recorder'], algorithm1['uplink_delay_recorder'], label='algorithm1')
    # axes[1, 1].plot(algorithm2['iter_recorder'], algorithm2['uplink_delay_recorder'], label='algorithm2')
    # axes[1, 1].legend()

    D = 3000 # 每个用户的数据集大小
    K = 1000000 # 处理一个样本所需的CPU周期
    L = 1  # 本地迭代次数
    F = 1 * 1e3 * 1e3 * 1e3 # 每个用户的CPU频率
    computing_delay = D * K * L / F # 计算延时

    random_training_time_cost = []
    for i, item in enumerate(random['uplink_delay_cost']):
        random_training_time_cost.append(item + computing_delay * (i + 1))

    alg1_training_time_cost = []
    for i, item in enumerate(algorithm1['uplink_delay_cost']):
        alg1_training_time_cost.append(item + computing_delay * (i + 1))

    alg2_training_time_cost = []
    for i, item in enumerate(algorithm2['uplink_delay_cost']):
        alg2_training_time_cost.append(item + computing_delay * (i + 1))

    f, axes = plt.subplots(1, 2)
    axes[1].plot(random_training_time_cost, random['loss_recorder'], '-.', label='Random', color=color3, markersize=4)
    axes[1].plot(alg1_training_time_cost, algorithm1['loss_recorder'], '--', label='Separated Optimization', color=color2, markersize=4)
    axes[1].plot(alg2_training_time_cost, algorithm2['loss_recorder'], '-', label='Joint Optimization', color=color1, markersize=4)
    axes[1].legend()
    axes[1].grid()
    axes[1].set_xlabel('Training Time (s)', fontdict=font)
    axes[1].set_ylabel('Loss', fontdict=font)
    axes[1].set_title('Comparison of the Convergence Speed of the three methods [Loss]', fontdict=font)


    axes[0].plot(random_training_time_cost, random['acc_recorder'], '-.', label='Random', color=color3, markersize=4)
    axes[0].plot(alg1_training_time_cost, algorithm1['acc_recorder'], '--', label='Separated Optimization', color=color2, markersize=4)
    axes[0].plot(alg2_training_time_cost, algorithm2['acc_recorder'], '-', label='Joint Optimization', color=color1, markersize=4)
    axes[0].legend()
    axes[0].grid()
    axes[0].set_xlabel('Training Time (s)', fontdict=font)
    axes[0].set_ylabel('Accuracy', fontdict=font)
    axes[0].set_title('Comparison of the Convergence Speed of the three methods [Accuracy]', fontdict=font)
    
    plt.show()

##############################################################################################################################

# if __name__ == '__main__':
#     random = np.load('./output/random_with_sche_info.npz', allow_pickle=True)
#     algorithm1 = np.load('./output/algorithm1_with_sche_info.npz', allow_pickle=True)
#     algorithm2 = np.load('./output/algorithm2_with_sche_info.npz', allow_pickle=True)
#     random_1 = np.load('./output/random_with_sche_info_1.npz', allow_pickle=True)
#     algorithm1_1 = np.load('./output/algorithm1_with_sche_info_1.npz', allow_pickle=True)
#     algorithm2_1 = np.load('./output/algorithm2_with_sche_info_1.npz', allow_pickle=True)
#     random_2 = np.load('./output/random_with_sche_info_2.npz', allow_pickle=True)
#     algorithm1_2 = np.load('./output/algorithm1_with_sche_info_2.npz', allow_pickle=True)
#     algorithm2_2 = np.load('./output/algorithm2_with_sche_info_2.npz', allow_pickle=True)

#     font = {
#         'size': 10
#     }

#     f, axes = plt.subplots(1, 2)

#     positions = random['schedualed_user_info'].item()
#     positions_1 = random_1['schedualed_user_info'].item()
#     positions_2 = random_2['schedualed_user_info'].item()
#     all_positions = []
#     for iter_info in positions.values():
#         for user_info in iter_info.values():
#             all_positions.append(user_info['position'])
#     for iter_info in positions_1.values():
#         for user_info in iter_info.values():
#             all_positions.append(user_info['position'])
#     for iter_info in positions_2.values():
#         for user_info in iter_info.values():
#             all_positions.append(user_info['position'])
#     position, counts = np.unique(np.round(all_positions, decimals=-1), axis=0, return_counts=True)
#     position /= 10
#     position[:, 0] += 10
#     position[:, 1] -= 10
#     position[:, 1] *= -1
#     print(position)
#     print(counts)
#     heatmap = np.zeros((21, 21))
#     for i, p in enumerate(position):
#         heatmap[int(p[0]), int(p[1])] = counts[i]
#     im = axes[0].imshow(heatmap / 3, vmin=0, vmax=10, cmap='Greys')
#     axes[0].set_xticks(np.arange(0, 21, 1))
#     axes[0].set_yticks(np.arange(0, 21, 1))
#     axes[0].set_xticklabels(np.arange(-100, 110, 10))
#     axes[0].set_yticklabels(np.arange(100, -110, -10))
#     plt.setp(axes[0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
#     thetas = np.linspace(0, np.math.pi*2, 200)
#     x = 10 * np.cos(thetas) + 10
#     y = 10 * np.sin(thetas) + 10
#     axes[0].plot(x, y, '--k')
#     axes[0].set_title('Heatmap of Scheduled User Location [Random]')
    
#     axes[0].figure.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

#     #############################33

#     positions = algorithm1['schedualed_user_info'].item()
#     positions_1 = algorithm1_1['schedualed_user_info'].item()
#     positions_2 = algorithm1_2['schedualed_user_info'].item()
#     all_positions = []
#     for iter_info in positions.values():
#         for user_info in iter_info.values():
#             all_positions.append(user_info['position'])
#     for iter_info in positions_1.values():
#         for user_info in iter_info.values():
#             all_positions.append(user_info['position'])
#     for iter_info in positions_2.values():
#         for user_info in iter_info.values():
#             all_positions.append(user_info['position'])
#     position, counts = np.unique(np.round(all_positions, decimals=-1), axis=0, return_counts=True)
#     position /= 10
#     position[:, 0] += 10
#     position[:, 1] -= 10
#     position[:, 1] *= -1
#     print(position)
#     print(counts)
#     heatmap = np.zeros((21, 21))
#     for i, p in enumerate(position):
#         heatmap[int(p[0]), int(p[1])] = counts[i]
#     im = axes[1].imshow(heatmap / 3, vmin=0, vmax=10, cmap='Greys')
#     axes[1].set_xticks(np.arange(0, 21, 1))
#     axes[1].set_yticks(np.arange(0, 21, 1))
#     axes[1].set_xticklabels(np.arange(-100, 110, 10))
#     axes[1].set_yticklabels(np.arange(100, -110, -10))
#     plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
#     thetas = np.linspace(0, np.math.pi*2, 200)
#     x = 10 * np.cos(thetas) + 10
#     y = 10 * np.sin(thetas) + 10
#     axes[1].plot(x, y, '--k')
#     axes[1].set_title('Heatmap of Scheduled User Location [Separated Optimization]')
    
#     axes[1].figure.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)




#     plt.show()
    
#######################################################################################################################

# def calculate_delay():
#     client_num = 20 # 用户数量
#     total_dataset_size = 60000 # 数据集总大小
#     dataset_size_per_client = total_dataset_size / client_num # 每个用户的数据集大小
#     D = dataset_size_per_client * np.ones((1, client_num)) # 每个用户的数据集大小
#     K = 100000 * np.ones((1, client_num)) # 处理一个样本所需的CPU周期
#     L = 1 * np.ones((1, client_num)) # 本地迭代次数
#     F = 2 * 1e3 * 1e3 * 1e3 * np.ones((1, client_num)) # 每个用户的CPU频率 2
#     computing_delay = D * K * L / F # 计算延时
    
#     M = 1024 * 1024 * 3 # 模型大小 bit
#     bandwidth_per_user = 50 * 1e3 * 1e3 # 频谱带宽 Hz
#     f = bandwidth_per_user * np.ones((1, client_num))
#     S = 1 * np.ones((1, client_num)) # 发送功率 1
#     N0 = 10**(-10) # 噪声功率谱密度
#     g = np.ones((1, client_num)) * (50**(-2)) # 信道增益
#     I = np.random.uniform(1e-4, 1e-2, (1, client_num)) # 干扰
#     rate = (f * np.log2(1 + g * S / (N0 * f + I))) # 速率
#     communication_delay = M / rate
#     print('***************')
#     print(rate / 8 / 1024 / 1024)
#     print(N0 * f)
#     print(I)
#     print(communication_delay, computing_delay)

#     total_delay = np.max(computing_delay + communication_delay)
#     return total_delay

# 变化计算频率，发射功率，频谱带宽

# if __name__ == '__main__':
#     f = np.arange(1, 41) / 10
#     delay_ = []
#     for item in f:
#         delay = calculate_delay(item)
#         delay_.append(delay)
    
#     font = {
#         'size': 10
#     }
#     plt.plot(f, delay_, 'o-', color=color1, label='Iteration Delay')
#     plt.xlabel('User Computing Frequency (GHz)', fontdict=font)
#     plt.ylabel('Iteration Delay (s)', fontdict=font)
#     plt.title('The Relationship between Iteration Delay and Computing Frequency', fontdict=font)
#     plt.legend()
#     plt.grid()
#     plt.show()

# if __name__ == '__main__':
#     # f = [10**(item) for item in range(-6, 2)]
#     f = [0.2*1e-5, 0.5*1e-5, 1e-5, 
#          0.2*1e-4, 0.5*1e-4, 1e-4, 
#          0.2*1e-3, 0.5*1e-3, 1e-3, 
#          0.2*1e-2, 0.5*1e-2, 1e-2, 
#          0.2*1e-1, 0.5*1e-1, 1e-1, 
#          0.2*1e0, 0.5, 1, 
#          0.2*1e1, 5, 10, 
#          0.2*1e2, 50, 1e2]
#     delay_ = []
#     for item in f:
#         delay = calculate_delay(item)
#         delay_.append(delay)
    
#     font = {
#         'size': 10
#     }
#     plt.loglog(f, delay_, 'o-', color=color1, label='Iteration Delay')
#     # plt.semilogx(f, delay_, 'o-', color=color1, label='Iteration Delay')
#     # plt.plot(f, delay_, 'o-', color=color1, label='Iteration Delay')
#     plt.xlabel('User Transmit Power (W)', fontdict=font)
#     plt.ylabel('Iteration Delay (s)', fontdict=font)
#     plt.title('The Relationship between Iteration Delay and Transmit Power', fontdict=font)
#     plt.legend()
#     plt.grid()
#     plt.show()

# if __name__ == '__main__':
#     f = np.arange(1, 51, 2)
#     delay_ = []
#     for item in f:
#         delay = calculate_delay(item)
#         delay_.append(delay)
    
#     font = {
#         'size': 10
#     }
#     plt.plot(f, delay_, 'o-', color=color1, label='Iteration Delay')
#     plt.xlabel('User Uplink Bandwidth (MHz)', fontdict=font)
#     plt.ylabel('Iteration Delay (s)', fontdict=font)
#     plt.title('The Relationship between Iteration Delay and Uplink Bandwidth', fontdict=font)
#     plt.legend()
#     plt.grid()
#     plt.show()

