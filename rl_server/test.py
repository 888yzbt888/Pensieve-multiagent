import multiprocessing
import time
from multiprocessing import Manager

def f():
    print time.ctime(),'subproc'

def work(d,l):
   while 1:
       l.append(time.ctime())
       f()
       time.sleep(10)

if __name__ == '__main__':
    manager=Manager()
    d=manager.dict()
    l=manager.list()
    p = multiprocessing.Process(target=work,args=(d,l))
    p.start()
    p.deamon = True
    while 1:
        print 'mainproc',l
        
        time.sleep(1)

'''
################################
#multiprocessing share variables
manager=multiprocessing.Manager()
Que1=manager.list()
Que2=manager.list()
QueOnline=manager.list()
################################
            elif('heartbeat' in post_data):
                global Que1
                Que1.append(self.client_address[0])
###### onlineCheck #######
def onlineCheck(Que1_,Que2_,QueOL):
    while True:
        #print('updateQue')
        updateQue(Que1_,Que2_,QueOL)
        time.sleep(5)
def updateQue(Que1_,Que2_,QueOL):
    #print('_Que1',Que1_[:])
    #print('_Que2',Que2_[:])
    #print('_QueOnline',QueOL[:])
    QueOL[:]=Que1_[:]+[item for item in Que2_[:] if item not in Que1_[:]]
    Que2_[:]=copy.copy(Que1_[:])
    Que1_[:]=[]
    #print('Que1_',Que1_[:])
    #print('Que2_',Que2_[:])
    print('QueOnline_',QueOL[:])
##########################
####### onlineCheck ######
        global Que1
        global Que2
        global QueOnline
        p = multiprocessing.Process(target=onlineCheck,args=(Que1,Que2,QueOnline))
        p.start()
        p.deamon = True
##########################
'''
