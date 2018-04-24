import json
import sys, getopt
opts,args=getopt.getopt(sys.argv[1:],"hc:")
cut_start=0
for op,value in opts:
    if op=="-c":
        cut_start=int(value)
    if op=="-h":
        print("-c cut start time (s) \nexample: $ python reward_summary -c 20")
        sys.exit()
f=open('ip_time_rewards.json','r')
load_rewards=json.load(f)
f.close()
r_overall=0
begin_time=0
end_time=0
for ip in load_rewards.keys():
    begin_time=int(min(load_rewards[ip].keys()))
    end_time=int(max(load_rewards[ip].keys()))
    print("ip: %s"% ip.encode('utf-8'))
    r_sum=0
    c_r_quantity=0
    for t in load_rewards[ip].keys():
        if int(t)>=begin_time+cut_start:
            r_sum+=load_rewards[ip][t]
        else:
            c_r_quantity+=1
    try:
        ip_reward=r_sum/(len(load_rewards[ip])-c_r_quantity)
        r_overall+=ip_reward
    except:
        print('devided by 0 1')
    print('reward= %.6f'% ip_reward)
try:
    r_overall/=len(load_rewards)
except:
    print('devided by 0 2')
print('')
print("experiment lasts %d - %d = %d s"% ((end_time-begin_time) , cut_start , (end_time-begin_time-cut_start)))
print('reward overall= %.6f'% r_overall)
