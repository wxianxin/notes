# Author: Steven Wang   Date: 20160702
# python 3.5.2

import socket

def main():
    host = '127.0.0.1'
    port = 1689
    #the port has to be different, because technically we are setting up another server
    server = ('127.0.0.1', 1688)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind((host, port))

    message = input('-> ')
    while message != 'q':
    	s.sendto(message.encode(), server)
    	data, addr = s.recvfrom(1024)
    	print('Received from server:' + data.decode('utf_8'))
    	message = input('-> ')
    s.close()

if __name__ == '__main__':
	main()