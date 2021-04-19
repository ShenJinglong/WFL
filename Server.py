
import numpy as np
import matplotlib.pyplot as plt

import params

from CommServer import CommServer
from TFTrainer import TFTrainer as TR

def aggregate_models(local_models):
    return np.mean(local_models, axis=0)

def select_based_on_weight(msg):
    grad_norm_sum = 0
    for item in msg.values():
        grad_norm_sum += item['grad_norm']
    select_probs = {id:msg[id]['grad_norm']/grad_norm_sum for id in msg.keys()}
    a = np.random.choice(list(select_probs.keys()), params.SCHEDUAL_SIZE, replace=False, p=list(select_probs.values()))
    sche_sig = {id:0 for id in msg.keys()}
    for id in a:
        sche_sig[id] = 1
    return sche_sig, a

def random_select():
    a = np.random.choice(range(params.CLIENT_NUM), params.SCHEDUAL_SIZE, replace=False, p=[1/params.CLIENT_NUM]*params.CLIENT_NUM)
    sche_sig = {id:0 for id in range(params.CLIENT_NUM)}
    for id in a:
        sche_sig[id] = 1
    return sche_sig, a

def get_uplink_delay(received_power):
    I = np.random.uniform(1e-4, 0.01, size=(params.RB_NUM,1))
    SINR = received_power / (I + params.UPLINK_BANDWIDTH * params.NOISE_POWER_SPECTRAL_DENSITY)
    rate = params.UPLINK_BANDWIDTH * np.log2(1 + SINR)
    delay = params.MODEL_SIZE / rate
    return delay

def get_RB_schedual(eng, msg, user_sche):
    import matlab

    received_power = []
    for id in range(params.CLIENT_NUM):
        received_power.append(msg[id]['received_power'])
    
    uplink_delay = get_uplink_delay(np.array([received_power]))

    # 构造不等式约束
    part_1 = []
    for i in range(params.CLIENT_NUM):
        part_1.append(np.identity(params.RB_NUM))
    part_1.append(np.zeros((params.RB_NUM, 1)))
    part_1 = np.concatenate(part_1, axis=1)
    part_2 = []
    for i in range(params.CLIENT_NUM):
        ma = np.zeros((params.CLIENT_NUM, params.RB_NUM))
        ma[i, :] = uplink_delay[:, i].transpose()
        part_2.append(ma)
    part_2.append(-np.ones((params.CLIENT_NUM, 1)))
    part_2 = np.concatenate(part_2, axis=1)
    A = matlab.double(np.concatenate([part_1, part_2], axis=0).tolist())
    b = matlab.double(np.concatenate([np.ones((params.RB_NUM, 1)), np.zeros((params.CLIENT_NUM, 1))], axis=0).tolist())

    # 构造等式约束
    part = []
    for i in range(params.CLIENT_NUM):
        ma = np.zeros((params.CLIENT_NUM, params.RB_NUM))
        ma[i,:] = np.ones((1, params.RB_NUM))
        part.append(ma)
    part.append(np.zeros((params.CLIENT_NUM, 1)))
    Aeq = matlab.double(np.concatenate(part, axis=1).tolist())
    a = []
    for i in range(params.CLIENT_NUM):
        a.append(user_sche[i])
    beq = matlab.double(np.array([a]).transpose().tolist())

    # 限定为 0-1 规划
    lb = matlab.double(np.zeros((params.CLIENT_NUM * params.RB_NUM + 1,)).tolist())
    ub = np.ones((params.CLIENT_NUM * params.RB_NUM + 1,))
    ub[-1] = np.Inf
    ub = matlab.double(ub.tolist())

    f = np.zeros((params.CLIENT_NUM * params.RB_NUM + 1, 1))
    f[-1,0] = 1
    f = matlab.double(f.tolist())
    intcon = matlab.double(list(range(1, params.CLIENT_NUM * params.RB_NUM + 1)))

    result = eng.intlinprog(f, intcon, A, b, Aeq, beq, lb, ub)
    allo_ma = np.array(result)[:-1].reshape((params.CLIENT_NUM, params.RB_NUM)).transpose()
    r_delay_ma = allo_ma * uplink_delay
    # print(np.max(r_delay_ma, axis=0))
    # print(result[-1])
    # print(np.argmax(r_delay_ma, axis=0))
    return np.argmax(r_delay_ma, axis=0), np.max(r_delay_ma, axis=0), result[-1]

def random_RB_schedual(msg, user_sche):
    received_power = []
    for id in range(params.CLIENT_NUM):
        received_power.append(msg[id]['received_power'])
    
    uplink_delay = get_uplink_delay(np.array([received_power]))

    _RB_sche = np.random.choice(range(params.RB_NUM), params.SCHEDUAL_SIZE, replace=False, p=[1/params.RB_NUM]*params.RB_NUM)
    RB_sche = []
    ii = 0
    for id in range(params.CLIENT_NUM):
        if user_sche[id] == 1:
            RB_sche.append(_RB_sche[ii])
            ii += 1
        else:
            RB_sche.append(0)
    user_delay = []
    for id in range(params.CLIENT_NUM):
        if user_sche[id] == 1:
            user_delay.append(uplink_delay[RB_sche[id], id])
        else:
            user_delay.append(0)
    return RB_sche, user_delay, max(user_delay)

def draw_circle(ax):
    thetas = np.linspace(0, np.math.pi*2, 200)
    x = 100 * np.cos(thetas)
    y = 100 * np.sin(thetas)
    ax.plot(x, y)

class Server():
    def __init__(self) -> None:
        pass

    def run(self, data, label):
        self.__comm = CommServer('127.0.0.1', 12345, params.CLIENT_NUM)
        self.__trainer = TR()

        import matlab.engine
        eng = matlab.engine.start_matlab()

        loss_recorder = []
        acc_recorder = []
        iter_recorder = []
        uplink_delay_recorder = []

        # plt.figure(figsize=(8, 6), dpi=80)
        f, axes = plt.subplots(3, 2)
        for ax in axes[:, 0]:
            ax.remove()
        axes = axes[:, 1]
        gs = axes[0].get_gridspec()
        axbig = f.add_subplot(gs[:, 0])
        plt.ion()
        
        for iter in range(params.ITERATION_NUM):
            print(f'************************************ ITERATION {iter} *****************************************')
            # 获取 global model 的权重
            self.__global_model = self.__trainer.get_weights()
            # 向每一个用户下发 global model
            print('delivering global model ...')
            self.__comm.broadcast(self.__global_model)
            # 接收每个用户的梯度范数
            msg = self.__comm.recv_all()    
            print('gradient received ...')
            # 基于梯度范数给出调度结果
            sche_sig, a = select_based_on_weight(msg)
            # sche_sig, a = random_select() # 随机调度
            # 给出资源分配结果
            RB_sche, user_delay, iter_delay = get_RB_schedual(eng, msg, sche_sig)
            # RB_sche, user_delay, iter_delay = random_RB_schedual(msg, sche_sig) # 随机分配
            # 下发调度结果
            print('sending schedual command ...')
            self.__comm.send(sche_sig)
            # 接收被调度用户的 local model
            local_models = self.__comm.recv(list(a))
            print('local models received ...')
            # 融合模型
            self.__trainer.aggregate(local_models)
            # 评估当前模型
            eval_result = self.__trainer.evaluate(data, label)

            loss_recorder.append(eval_result[0])
            acc_recorder.append(eval_result[1])
            iter_recorder.append(iter)
            uplink_delay_recorder.append(iter_delay)
            user_position_recorder = []
            for i in range(params.CLIENT_NUM):
                user_position_recorder.append(msg[i]['position'])

            axes[0].clear()            
            axes[0].set_xlim(0, params.ITERATION_NUM)
            axes[0].set_ylim(0, 1)
            axes[0].plot(iter_recorder, acc_recorder, label='acc')
            axes[0].legend()
            axes[1].clear()
            axes[1].set_xlim(0, params.ITERATION_NUM)
            axes[1].set_ylim(0, 3)        
            axes[1].plot(iter_recorder, loss_recorder, label='loss')
            axes[1].legend()
            axes[2].clear()
            axes[2].set_xlim(0, params.ITERATION_NUM)
            axes[2].plot(iter_recorder, uplink_delay_recorder, label='uplink delay')
            axes[2].plot([0, params.ITERATION_NUM], [np.mean(uplink_delay_recorder)]*2, label='mean value')
            axes[2].text(int(params.ITERATION_NUM / 2), np.mean(uplink_delay_recorder) + 2, f'{np.mean(uplink_delay_recorder):.2f}')
            axes[2].legend()
            axbig.clear()
            axbig.axis('equal')
            draw_circle(axbig)
            for i, p in enumerate(user_position_recorder):
                axbig.scatter(p[0], p[1])
                if i in a:
                    axbig.plot([0, p[0]], [0, p[1]])
                
            plt.pause(0.0001)

        plt.ioff()
        plt.show()

        eng.exit()

if __name__ == '__main__':
    # server = Server()
    # server.run()
    # eng = matlab.engine.start_matlab()
    # get_RB_schedual(eng, [], [])
    # eng.exit()
    pass