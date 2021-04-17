import socket
import pickle
import numpy as np

class CommClient():
    def __init__(self, host, port) -> None:
        self.__sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.__sock.connect((host, int(port)))

    def recv(self):
        return pickle.loads(self.__sock.recv(1024 * 1024))

    def send(self, data):
        return self.__sock.sendall(pickle.dumps(data))

if __name__ == '__main__':
    comm_client1 = CommClient('127.0.0.1', 12345)
    comm_client2 = CommClient('127.0.0.1', 12345)
    comm_client3 = CommClient('127.0.0.1', 12345)
    comm_client4 = CommClient('127.0.0.1', 12345)
    comm_client5 = CommClient('127.0.0.1', 12345)
    comm_client6 = CommClient('127.0.0.1', 12345)
    print(repr(comm_client1.recv()))
    print(repr(comm_client2.recv()))
    print(repr(comm_client3.recv()))
    print(repr(comm_client4.recv()))
    print(repr(comm_client5.recv()))
    print(repr(comm_client6.recv()))
    input()
    comm_client1.send(0)
    comm_client2.send(2)
    comm_client3.send(0)
    comm_client4.send(4)
    comm_client5.send(7)
    comm_client6.send(100)
    input()
    print(repr(comm_client1.recv()))
    print(repr(comm_client2.recv()))
    print(repr(comm_client3.recv()))
    print(repr(comm_client4.recv()))
    print(repr(comm_client5.recv()))
    print(repr(comm_client6.recv()))
    input()
    comm_client1.send(np.random.random((5,5)))
    comm_client2.send(np.random.random((5,5)))
    comm_client3.send(np.random.random((5,5)))
    comm_client4.send(np.random.random((5,5)))
    comm_client5.send(np.random.random((5,5)))
    comm_client6.send(np.random.random((5,5)))
