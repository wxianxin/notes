# Author: Steven Wang   Date: 20160702
# python 3.5.2

import socket

def main():
    host = '127.0.0.1'
    port = 1688
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind((host, port))
    
    print('Server Started.')
    while True:
    	data, addr = s.recvfrom(1024)
    	#udp
    	print("message from: " + str(addr))
    	print("data from connection: " + data.decode("utf-8"))
    	data = data.decode("utf-8").upper()
    	print('sending: ' + data)
    	s.sendto(data.encode(), addr)
    	#udp, we need to remember the incoming addr so that we can send back
    s.close()
if __name__ == '__main__':
	main()