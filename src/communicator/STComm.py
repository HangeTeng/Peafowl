import SSS
import numpy as np


class Sender(SSS.OSNSender):
    def __init__(self, size, permute, p=1<<128, ios_threads = 4 ):
        super().__init__(size=size, dest=permute,  p=self.__split_largeint(p), ios_threads = ios_threads) 

    def run(self, size,  p=1<<128, Sip="127.0.0.1:12222", ot_type=1, sessionHint="",num_threads=1):
        result = []
        a_list = super().run(size=size,   p=self.__split_largeint(p), Sip=Sip, ot_type=ot_type,sessionHint=sessionHint, num_threads=num_threads)
        for a in a_list:
            result.append(a[0] + (a[1]<<64))
        return np.array(result, dtype=object)

    def __split_largeint(self, largeint):
        arr_split = []
        temp = largeint
        while temp > 0:
            arr_split.append(temp & 0xFFFFFFFFFFFFFFFF)  # Extract the lowest 64 bits
            temp >>= 64  # Shift the input integer to the right by 64 bits
        return arr_split
    
class Receiver(SSS.OSNReceiver):
    def run(self, size, p=1<<128, Sip="127.0.0.1:12222", ot_type=1, sessionHint="", num_threads=1):
        result_1 = []
        result_2 = []
        a_list = super().run(size=size, p=self.__split_largeint(p), Sip=Sip, ot_type=ot_type, sessionHint=sessionHint,num_threads=num_threads)
        for a in a_list[0]:
            result_1.append(int(a[0] + (a[1]<<64)))
        for a in a_list[1]:
            result_2.append(int(a[0] + (a[1]<<64)))
        return np.array([result_1,result_2], dtype=object)

    def __split_largeint(self, largeint):
        arr_split = []
        temp = largeint
        while temp > 0:
            arr_split.append(temp & 0xFFFFFFFFFFFFFFFF)  # Extract the lowest 64 bits
            temp >>= 64  # Shift the input integer to the right by 64 bits
        return arr_split

if __name__ == "__main__":
    import multiprocessing

    size = 60
    p =1<<128
    num_threads = 1
    # Sip = "127.0.0.1:12222"
    # dest = range(size)
    permute = range(size-1, -1, -1)

    # receiver = Receiver(4)
    # print("receiver!")
    # sender = Sender(4)
    # print("Sender!")

    
    def receiver_process(result_queue, sessionHint, Sip):
        receiver = Receiver(4)
        print("receiver_process!")
        print("receiver total sent"+str(receiver.getTotalDataSent()))
        a = receiver.run(size=size, sessionHint=sessionHint, p=p, Sip=Sip, num_threads=num_threads)
        print(a.dtype)
        print("receiver run!")
        result_queue.put(a)

    

    def sender_process(result_queue, sessionHint, Sip):
        sender = Sender(size=size, permute=permute,p=p,)
        a = sender.run(size=size, sessionHint=sessionHint, p=p, Sip=Sip, num_threads=num_threads)
        print("sender run!")
        result_queue.put(a)

    result_queue_recv = multiprocessing.Queue()
    result_queue_send = multiprocessing.Queue()
    result_queue_recv2 = multiprocessing.Queue()
    result_queue_send2 = multiprocessing.Queue()
    
    
    receiver_proc = multiprocessing.Process(target=receiver_process, args=(result_queue_recv,"1","127.0.0.1:12246"))
    sender_proc = multiprocessing.Process(target=sender_process, args=(result_queue_send,"1","127.0.0.1:12246"))
    receiver_proc2 = multiprocessing.Process(target=receiver_process, args=(result_queue_recv2,"2","127.0.0.1:12296"))
    sender_proc2 = multiprocessing.Process(target=sender_process, args=(result_queue_send2,"2","127.0.0.1:12296"))
    

    
    print("start!")
    receiver_proc.start()
    sender_proc.start()
    receiver_proc2.start()
    sender_proc2.start()
    
    receiver_proc.join()
    sender_proc.join()
    receiver_proc2.join()
    sender_proc2.join()


    print("send get")
    send = result_queue_send.get()
    print("recv get")


    
    recv = result_queue_recv.get()
    
    print("checking...")
    for i in range(size):
        print(send[i])
        print(recv[1][i])
        print(recv[0][i])
        print(p)
        if((send[i]+recv[1][i])% p != recv[0][permute[i]]):
            print("error!")
            break
    if i == size-1:
        print("correct!")