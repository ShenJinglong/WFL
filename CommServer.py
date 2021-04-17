
import selectors
import socket
import types
import pickle

def establish_connection(sel, sock, id):
    conn, addr = sock.accept()
    print('accepted connection from', addr, id)
    conn.setblocking(False)
    data = types.SimpleNamespace(addr=addr, id=id, inb=b'', outb=b'')
    events = selectors.EVENT_READ | selectors.EVENT_WRITE
    sel.register(conn, events, data=data)

class CommServer():
    def __init__(self, host, port, num_conns) -> None:
        self.__sel = selectors.DefaultSelector()

        lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        lsock.bind((host, int(port)))
        lsock.listen()
        lsock.setblocking(False)
        self.__sel.register(lsock, selectors.EVENT_READ, data=None)

        while len(self.__sel.get_map()) < num_conns + 1:
            events = self.__sel.select(timeout=None)
            for key, mask in events:
                if key.data is None:
                    establish_connection(self.__sel, key.fileobj, len(self.__sel.get_map())-1)

        self.__num_conns = num_conns
        print('initialization finished ...')

    def broadcast(self, data):
        target_user = list(range(len(self.__sel.get_map())-1))
        while len(target_user) != 0:
            events = self.__sel.select(timeout=None)
            for key, mask in events:
                if mask & selectors.EVENT_WRITE:
                    if key.data.id in target_user:
                        key.fileobj.sendall(pickle.dumps(data))
                        target_user.remove(key.data.id)

    def send(self, data):
        while len(data) != 0:
            events = self.__sel.select(timeout=None)
            for key, mask in events:
                if mask & selectors.EVENT_WRITE:
                    if key.data.id in data.keys():
                        key.fileobj.sendall(pickle.dumps(data[key.data.id]))
                        del data[key.data.id]

    def recv_all(self):
        result = {}
        while len(result) != self.__num_conns:
            events = self.__sel.select(timeout=None)
            for key, mask in events:
                if mask & selectors.EVENT_READ:
                    if key.data is not None:
                        if key.data.id not in result.keys():
                            result[key.data.id] = pickle.loads(key.fileobj.recv(1024 * 1024))
        return result

    def recv(self, index):
        result = {}
        while len(index) != 0:
            events = self.__sel.select(timeout=None)
            for key, mask in events:
                if mask & selectors.EVENT_READ:
                    if key.data is not None:
                        if key.data.id in index:
                            result[key.data.id] = pickle.loads(key.fileobj.recv(1024 * 1024))
                            index.remove(key.data.id)
        return result
    
if __name__ == '__main__':
    comm_server = CommServer('127.0.0.1', 12345, 6)
    # comm_server.broadcast(b'hello world!!!')
    # comm_server.send({
    #     0: b'hi',
    #     1: b'hello',
    #     2: b'how are you',
    #     3: b'message',
    #     5: b'jinglong'
    # })

    print(comm_server.recv([0, 1, 5, 3]))