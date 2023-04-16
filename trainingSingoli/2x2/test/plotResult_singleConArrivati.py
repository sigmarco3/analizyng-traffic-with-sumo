import sys
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
def plotResult(file,file2):

        df = pd.read_csv(file + '_conn0_run1.csv')  #leggo i file con il training copiato


        x = df['step']
        t =df['total_vehicle']
        y = df['system_total_waiting_time']
        z = df['system_total_stopped']
        u = df['system_mean_waiting_time']
        v = df['system_mean_speed']
        a = df['arrived']
        s = df['loaded']

        df1 = pd.read_csv(file + '_conn0_run2.csv')
        y1 = df['system_total_waiting_time']


        t1 = df1['total_vehicle']
        z1 = df1['system_total_stopped']
        u1 = df1['system_mean_waiting_time']
        v1 = df1['system_mean_speed']
        a1 = df['arrived']
        s1 = df['loaded']
        df2 = pd.read_csv(file + '_conn0_run3.csv')
        y2 = df2['system_total_waiting_time']
        z2 = df2['system_total_stopped']
        u2 = df2['system_mean_waiting_time']
        v2 = df2['system_mean_speed']
        a2 = df['arrived']
        s2 = df2['loaded']

        t2 = df2['total_vehicle']
        df3 = pd.read_csv(file + '_conn0_run4.csv')
        y3 = df3['system_total_waiting_time']
        z3 = df3['system_total_stopped']
        u3 = df3['system_mean_waiting_time']
        v3 = df3['system_mean_speed']
        a3 = df['arrived']
        s3= df3['loaded']

        t3 = df3['total_vehicle']
        ym1 = (y + y1 + y2 + y3) / 4
        zm1 = (z + z1 + z2 + z3) / 4
        tm1 = (t + t1 +t2 +t3)/4
        um1 = (u + u1 + u2 + u3) / 4
        vm1 = (v + v1 + v2 + v3) / 4
        sm1 = (s + s1 +s2 +s3)/4 #media veicoli caricati
        am1 = (a+a1+a2+a3)/4 #media veicoli arrivati

        #plt.ylabel("system_total_waiting time")


        #plt.xlabel("step")
        #plt.title("rete 4x4 reward diversi tra colonne ")
       # plt.plot(x, ym)


        # mu1 = ym.mean()
        # sigma1 = ym.std()


        df = pd.read_csv(file2 + '_conn0_run1.csv')  #leggo i file con il training singolo
        # definiamo le dimensioni della finestra in pollici ed il dpi

        x = df['step']
        t = df['total_vehicle']
        y = df['system_total_waiting_time']
        z = df['system_total_stopped']
        u = df['system_mean_waiting_time']
        v = df['system_mean_speed']
        a = df['arrived']
        s = df['loaded']

        df1 = pd.read_csv(file2 + '_conn0_run2.csv')
        y1 = df['system_total_waiting_time']

        t1 = df1['total_vehicle']
        z1 = df1['system_total_stopped']
        u1 = df1['system_mean_waiting_time']
        v1 = df1['system_mean_speed']
        a1 = df['arrived']
        s1 = df['loaded']
        df2 = pd.read_csv(file2 + '_conn0_run3.csv')
        y2 = df2['system_total_waiting_time']
        z2 = df2['system_total_stopped']
        u2 = df2['system_mean_waiting_time']
        v2 = df2['system_mean_speed']
        a2 = df['arrived']
        s2 = df2['loaded']

        t2 = df2['total_vehicle']
        df3 = pd.read_csv(file2 + '_conn0_run4.csv')
        y3 = df3['system_total_waiting_time']
        z3 = df3['system_total_stopped']
        u3 = df3['system_mean_waiting_time']
        v3 = df3['system_mean_speed']
        a3 = df['arrived']
        s3= df3['loaded']

        t3 = df3['total_vehicle']
        ym2 = (y + y1 + y2 + y3) / 4
        zm2 = (z + z1 + z2 + z3) / 4   #total stopped secondo file
        tm2 = (t + t1 + t2 + t3) / 4   #medio auto circolanti
        um2 = (u + u1 + u2 + u3) / 4   #tempo medio di attesa
        vm2 = (v + v1 + v2 + v3) / 4   #velocità media secondo file
        am2 = (a +a1 +a2+a3)/4    #numero veicoli arrivati a destinazione
        sm2 = (s +s1+s2+s3)/4 #numero auto caricate
        # create chart


        # std_error = np.std(sm, ddof=1) / np.sqrt(len(sm))
        #
        # plt.bar(x=tm,height=sm, yerr=std_error)
        # plt.xlabel("auto caricate")
        # plt.ylabel("auto in circolazione")
        # plt.title("auto in circolazione - in coda nel caso di training singolo")
        # plt.show()

        m_v=np.arange(10)  #medie veicoli ogni 1000 secondi primo file
        zc1 = np.arange(10) #medie total stopped ogni 1000 secondi primo file
        zc2 = np.arange(10) #medie total stopped ogni 1000 secondi secondo file
        uc2 = np.arange(10)  #vettore valor medio waiting time ogni 1000 secondi secondo file
        uc1 = np.arange(10) #medie mean waiting time ogni 1000 secondi primo file
        vc1 = np.arange(10) #medie mean speed ogni 1000 secondi primo file
        vc2 = np.arange(10)
        ar1 = np.arange(10) #medie veicoli arrivati ogni 1000 secondi primo file
        ar2= np.arange(10) #medie veicoli arrivati ogni 1000 secondi secondo file
        sc1 = np.arange(10)
        sc2 = np.arange(10)
        time = np.arange(1000,11000,1000)
        for i in range(0,len(m_v)):
            m_v[i]=np.mean(tm1[(i*200):((i+1)*200)])
            zc1[i]=np.mean(zm1[(i * 200):((i + 1) * 200)])
            uc1[i] = np.mean(um1[(i * 200):((i + 1) * 200)])
            vc1[i] = np.mean(vm1[(i * 200):((i + 1) * 200)])
            ar1[i] = np.mean(am1[(i * 200):((i + 1) * 200)])
            sc1[i] = np.mean(sm1[(i * 200):((i + 1) * 200)])

        # create chart
        z1 = zc1 / m_v


        std_error = np.std(z1, ddof=1) / np.sqrt(len(z1))
        print(std_error)

        plt.errorbar(time,z1,yerr=std_error,marker="o",color="blue",markersize=2, label="reward fissa")  #auto ferme su auto circolanti in funzione del tempo
        print("valor medio di total stopped :", zc1.mean())
        print("massimo di total stopped :" , zc1.max(), "su max veicoli :", m_v.max())
        plt.xlabel("time-step",size=12)

        plt.title("rete 4x4 reward 'wait' e varie reward misura total stopped/running veicles ")
        plt.ylabel("system total stopped/running vehicles",size=12)





        #secondo grafico medie auto in 1000 secondi
        m_veh = np.arange(10)
        for i in range(0, len(m_veh) ):
            m_veh[i] = np.mean(tm2[(i * 200):((i + 1) * 200)])

            zc2[i]=np.mean(zm2[(i * 200):((i + 1) * 200)])
            uc2[i] = np.mean(um2[(i * 200):((i + 1) * 200)])
            vc2[i] = np.mean(vm2[(i * 200):((i + 1) * 200)])
            ar2[i] = np.mean(am2[(i * 200):((i + 1) * 200)])
            sc2[i] = np.mean(sm2[(i * 200):((i + 1) * 200)])

        z2 = zc2 / m_veh
        std_error = np.std(z2, ddof=1) / np.sqrt(len(z2))
        autoMedie = ((m_v + m_veh)/2).astype(int)
        m_v = m_v.astype(int)
        m_veh = m_veh.astype(int)
        #autoMedie = np.concatenate((m_v,m_veh),axis=0)
        print(autoMedie)
        print(m_veh)
        plt.errorbar(time,z2,yerr=std_error,marker="x",color="orange",markersize=2, label="varie reward")

        print("valor medio di total stopped :", zc2.mean())
        print("massimo di total stopped :", zc2.max(),"su max veicoli : ", m_veh.max())
        tick=np.arange(0,1000,100)
        plt.xticks(time, time)
        plt.legend()
        plt.grid()
        plt.show()

       #throughtput
        print("arrivati : " ,sm2)
        tput1 = ar1/sc1
        std_error = np.std(tput1, ddof=1) / np.sqrt(len(tput1))
        plt.errorbar(time, tput1, yerr=std_error, marker="o", color="blue", markersize=5, label="reward wait")
        tput2 = ar2/sc2
        std_error = np.std(tput2, ddof=1) / np.sqrt(len(tput2))
        plt.errorbar(time, tput2, yerr=std_error, marker="x", color="orange", markersize=5, label="reward diverse")
        plt.xlabel("time-step", size=12)
        plt.xticks(time, time)
        plt.title("rete 4x4 reward wait e varie reward misura troughput")
        plt.ylabel("system throughput", size=12)
        plt.legend()
        plt.grid()
        plt.show()
        #mean waiting time

        std_error = np.std(um1, ddof=1) / np.sqrt(len(um1))
        # create chart
        plt.errorbar(time,uc1,yerr=std_error,marker="o",color="blue",markersize=2, label="reward uguali")
        print("valor medio di mean waiting time :" ,uc1.mean())
        # for x,y in zip (m_v,uc1):
        #     label = int(x)
        #     plt.annotate(label,  # this is the text
        #                  (x, y),  # these are the coordinates to position the label
        #                  textcoords="offset points",  # how to position the text
        #                  xytext=(0, 5),  # distance from text to points (x,y)
        #                  ha='center')  # horizontal alignment can be left, right or center
        print("massimo di mean waiting time :", uc1.max())
        std_error = np.std(um2, ddof=1) / np.sqrt(len(um2))
        # create chart
        plt.errorbar(time,uc2,yerr=std_error,marker="x",color="orange",markersize=2,label="reward diverse")

        print("valor medio di mean waiting time :" ,uc2.mean())


        print("massimo di mean waiting time :", uc2.max())
        plt.xlabel("numero auto medio in circolazione ogni 1000 secondi",size=12)
        plt.title("rete 4x4 misura mean waiting time ")
        plt.ylabel("system mean waiting time(seconds)",size=12)
        plt.xticks(time,time)
        plt.legend()

        plt.grid()
        plt.show()
        #plt.plot(x, vm)

        #velocità media





        std_error = np.std(vm1, ddof=1) / np.sqrt(len(vm1))
        # create chart
        plt.errorbar(time, vc1, yerr=std_error, marker="o",markersize=2, color="blue", label="reward uguali")


        # for x, y in zip(m_v, vc1):
        #     label = int(x)
            # plt.annotate(label,  # this is the text
            #              (x, y),  # these are the coordinates to position the label
            #              textcoords="offset points",  # how to position the text
            #              xytext=(0, 5),  # distance from text to points (x,y)
            #              ha='center')  # horizontal alignment can be left, right or center
        std_error = np.std(vm2, ddof=1) / np.sqrt(len(vm2))
        plt.errorbar(time, vc2, yerr=std_error, marker="x", color="orange", markersize=2,label="reward diverse")
        plt.xlabel("time-step ")
        plt.title("rete 4x4 misura mean speed")
        plt.legend()
        plt.xticks(time, time)
        plt.grid()
        plt.ylabel("system mean speed(m/s)")
        plt.show()
if __name__ == '__main__':
        file2 = 'D:/programmi/sumo/esperimenti semafori/Reinforcement-learning-in-traffic-light/outputs/4x4/ql-4x4grid-variReward-10000sec(e auto)'
        plotResult('D:/programmi/sumo/esperimenti semafori/Reinforcement-learning-in-traffic-light/outputs/4x4/ql-4x4grid-unareward',file2)