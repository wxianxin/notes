# Author: Steven Wang   Date: 20160702
# python 3.5.2

import socket

def main():
    host = '127.0.0.1'
    port = 1688
    s = socket.socket()
    s.bind((host, port))
    s.listen(1)
    c, addr = s.accept()
    # s.accept() will return a new socket
    # c is current connection
    print ("connection from: " + str(addr))
    while True:
        data = c.recv(1024)
        #buffer is 1024 bytes
        if not data:
            break
        print ("data received from the connection: " + str(data))
        data = str(data.decode("utf-8")).upper()
        print ("sending: " + str(data))
        c.send(data.encode())
    c.close()
if __name__ == '__main__':
    main()
