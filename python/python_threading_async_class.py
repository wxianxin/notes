# Author: Steven Wang    Date:20160704
# python 3.5.2
import threading
import time

class AsyncWrite(threading.Thread):
	def __init__(self, text, out):
		threading.Thread.__init__(self)
		self.text = text
		self.out = out

	def run(self):
		f = open(self.out, 'a')
		f.write(self.text + '\n')
		f.close()
		time.sleep(2)
		print("Finished Background file write to " + self.out)

def main():
		message = input("Enter a string to store: ")
		background = AsyncWrite(message, "out.txt")
		background.start()
		print("THe program can continue to run while it writes in another thread")
		print(100+400)

		background.join()
		#This is a method from the class "Tread"; The main process will wait at this line until that thread is finished
		print("Waited until thread was complete")
	
if __name__ == "__main__":
		main()
