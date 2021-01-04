from threading import Thread
import globalDefs
import socket
import time

class serverThread(Thread):
    def ReconnectLoop(self):
        print('Server thread: Waiting for connection')
        self.conn = None
        while not self.conn and globalDefs.aliveFlag:
            try:
                self.conn, self.addr = self.Socket.accept()
                print('Server thread: Connection from ', self.addr)
                self.conn.settimeout(0.1)
            except socket.timeout:
                time.sleep(0.01)

    def __init__(self, im_width = 640, im_height = 480):
        Thread.__init__(self)
        self.Socket = socket.socket()
        self.Socket.bind(('', 9090))
        self.Socket.listen(1)
        self.Socket.settimeout(0.1)
        self.conn = None
        print('Server thread: initialized')

    def run(self):
        while globalDefs.aliveFlag:
            if self.conn:
                try:
                    c = self.conn.recv(64)
                    if c:
                        print('Server thread: ' + c.decode() + ' was received')
                    if c == b' ':
                        if globalDefs.cardNum:
                            card_id = globalDefs.cardNum
                            color_id = globalDefs.cardCID
                            if card_id == 10:
                                card_id2send = 0
                            else:
                                card_id2send = card_id
                            if color_id == 0:
                                 toSend = str(card_id2send) + 'r'
                            elif color_id == 1:
                                toSend = str(card_id2send) + 'b'
                            elif color_id == 2:
                                toSend = str(card_id2send) + 'g'
                            elif color_id == 3:
                                 toSend = str(card_id2send) + 'y'
                            elif color_id == 4:
                                toSend = str(card_id2send) + 'o'
                        else:
                            toSend = 'Empty'
                            print('Server thread: '+toSend+' was sent')
                        self.conn.send(toSend.encode())
                    if c == b'l':
                            self.conn.send(str(globalDefs.lineVal).encode())
                            print('Server thread: ' + str(globalDefs.lineVal) + ' was sent')
                except socket.timeout:
                    print('Server thread: SocketTimeout')
                    self.ReconnectLoop()
                except ConnectionResetError:
                    print('Server thread: ConnectionResetError')
                    self.ReconnectLoop()
            else:
                self.ReconnectLoop()






