# Author: Steven Wang   Date: 20160702
# python 3.5.2

import socket

def main():
    host = '127.0.0.1'
    port = 1688
    
    s = socket.socket()
    s.connect((host, port))
    
    message = input("-> ")
    while message != 'q':
        s.send(message.encode())
        data = s.recv(1024)
        print ('Received from server: ' + data.decode("utf-8"))
        message = input("-> ")
    s.close()

if __name__ == '__main__':
    main()