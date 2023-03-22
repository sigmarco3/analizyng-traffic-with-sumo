import sys
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
def plotResult(file,file2):
    #with open(file + '_conn0_run4.csv', newline="", encoding="ISO-8859-1") as filecsv:
        # lettore = csv.reader(filecsv, delimiter=",")
        # header = next(lettore)
        #
        # t = [(linea[0], linea[3]) for linea in lettore]
        # t = np.array(t)
        # dati = t[:, 1]
        #
        # time = t[:, 0]
        # times = np.array(time)
        # dati = [float(s) for s in dati]
        #
        # # print(datis.shape)
        #
        # # fig = plt.figure()
        # # ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        # plt.xlabel('secondi')
        # plt.ylabel('system total waiting time')
        # max = float(time[len(time)-1])
        #
        # plt.xticks(time)
        # plt.plot(time, dati, color='blue')
        #
        # plt.show()
        df = pd.read_csv(file + '_conn0_run1.csv')
        # definiamo le dimensioni della finestra in pollici ed il dpi
        from matplotlib.pyplot import figure
        figure(figsize=(18, 10), dpi=80)
        x = df['step']
        t =df['total_vehicle']
        y = df['system_total_waiting_time']
        z = df['system_total_stopped']
        u = df['system_mean_waiting_time']
        v = df['system_mean_speed']
        #s = df['loaded']

        df1 = pd.read_csv(file + '_conn0_run2.csv')
        y1 = df['system_total_waiting_time']


        t1 = df1['total_vehicle']
        z1 = df1['system_total_stopped']
        u1 = df1['system_mean_waiting_time']
        v1 = df1['system_mean_speed']
        #s1 = df['loaded']
        df2 = pd.read_csv(file + '_conn0_run3.csv')
        y2 = df2['system_total_waiting_time']
        z2 = df2['system_total_stopped']
        u2 = df2['system_mean_waiting_time']
        v2 = df2['system_mean_speed']
        #s2 = df2['loaded']

        t2 = df2['total_vehicle']
        df3 = pd.read_csv(file + '_conn0_run4.csv')
        y3 = df3['system_total_waiting_time']
        z3 = df3['system_total_stopped']
        u3 = df3['system_mean_waiting_time']
        v3 = df3['system_mean_speed']
        #s3= df3['loaded']

        t3 = df3['total_vehicle']
        ym1 = (y + y1 + y2 + y3) / 4
        zm1 = (z + z1 + z2 + z3) / 4
        tm1 = (t + t1 +t2 +t3)/4
        um1 = (u + u1 + u2 + u3) / 4
        vm1 = (v + v1 + v2 + v3) / 4
        #sm = (s + s1 +s2 +s3)/4


        #plt.ylabel("system_total_waiting time")


        #plt.xlabel("step")
        #plt.title("rete 4x4 reward diversi tra colonne ")
       # plt.plot(x, ym)


        # mu1 = ym.mean()
        # sigma1 = ym.std()


        df = pd.read_csv(file2 + '_conn0_run1.csv')  #leggo i file con il training singolo
        # definiamo le dimensioni della finestra in pollici ed il dpi
        from matplotlib.pyplot import figure

        figure(figsize=(18, 10), dpi=80)
        x = df['step']
        t = df['total_vehicle']
        y = df['system_total_waiting_time']
        z = df['system_total_stopped']
        u = df['system_mean_waiting_time']
        v = df['system_mean_speed']
        # s = df['loaded']

        df1 = pd.read_csv(file2 + '_conn0_run2.csv')
        y1 = df['system_total_waiting_time']

        t1 = df1['total_vehicle']
        z1 = df1['system_total_stopped']
        u1 = df1['system_mean_waiting_time']
        v1 = df1['system_mean_speed']
        # s1 = df['loaded']
        df2 = pd.read_csv(file2 + '_conn0_run3.csv')
        y2 = df2['system_total_waiting_time']
        z2 = df2['system_total_stopped']
        u2 = df2['system_mean_waiting_time']
        v2 = df2['system_mean_speed']
        # s2 = df2['loaded']

        t2 = df2['total_vehicle']
        df3 = pd.read_csv(file2 + '_conn0_run4.csv')
        y3 = df3['system_total_waiting_time']
        z3 = df3['system_total_stopped']
        u3 = df3['system_mean_waiting_time']
        v3 = df3['system_mean_speed']
        # s3= df3['loaded']

        t3 = df3['total_vehicle']
        ym2 = (y + y1 + y2 + y3) / 4
        zm2 = (z + z1 + z2 + z3) / 4
        tm2 = (t + t1 + t2 + t3) / 4
        um2 = (u + u1 + u2 + u3) / 4
        vm2 = (v + v1 + v2 + v3) / 4
        # create chart


        # std_error = np.std(sm, ddof=1) / np.sqrt(len(sm))
        #
        # plt.bar(x=tm,height=sm, yerr=std_error)
        # plt.xlabel("auto caricate")
        # plt.ylabel("auto in circolazione")
        # plt.title("auto in circolazione - in coda nel caso di training singolo")
        # plt.show()

        m_v=np.arange(10)  #medie veicoli ogni 1000 secondi primo file
        zc1 = np.arange(10)
        zc2 = np.arange(10)
        uc2 = np.arange(10)  #vettore valor medio media ogni 1000 secondi
        uc1 = np.arange(10)
        vc1 = np.arange(10)
        vc2 = np.arange(10)
        for i in range(0,len(m_v)):
            m_v[i]=np.mean(tm1[(i*200):((i+1)*200)])
            zc1[i]=np.mean(zm1[(i * 200):((i + 1) * 200)])
            uc1[i] = np.mean(um1[(i * 200):((i + 1) * 200)])
            vc1[i] = np.mean(vm1[(i * 200):((i + 1) * 200)])
        std_error = np.std(ym1, ddof=1) / np.sqrt(len(ym1))
        plt.errorbar(tm1,ym1,yerr=std_error,marker="o",color="blue")
        # # ax.set_ylabel("total waiting time")
        # # ax.grid()
        plt.title('2x2 wait  misura total waiting time')
        plt.xlabel('auto')
        plt.ylabel('system total waiting time(seconds)')
        plt.title("traffico crescente reward wait training indipendente")
        plt.show()
        # fig= plt.figure()
        # ax = fig.add_subplot(projection='3d')
        #
        #
        # ax.set_title("percorsi random reward queue training copiato")
        # #plt.plot(x,zm)
        # ax.set_xlabel("step")
        # ax.set_ylabel("auto")
        # ax.set_zlabel("system total waiting time(seconds")
        # x1 = x.values
        # y1 = tm.values
        # z1 = ym.values
        # ax.scatter(x1,y1,z1,alpha=1)
        std_error = np.std(zm1, ddof=1) / np.sqrt(len(zm1))
        # create chart
        width = 7
        ind = np.arange(10)
        plt.errorbar(m_v,zc1,yerr=std_error,marker="o",color="blue",markersize=2,label="training copiato")
        print("valor medio di total stopped :", zc1.mean())
        print("massimo di total stopped :" , zc1.max(), "su max veicoli :", m_v.max())
        plt.xlabel("numero auto medio in circolazione ogni 1000 secondi",size=12)

        plt.title("rete 4x4 reward wait traffico crescente misura total stopped ")
        plt.ylabel("system total stopped (vehicles)",size=12)


        std_error = np.std(zm2, ddof=1) / np.sqrt(len(zm2))
        #secondo grafico medie auto in 1000 secondi
        m_veh = np.array(tm2[0:2000:200])
        for i in range(0, len(m_veh) ):
            m_veh[i] = np.mean(tm2[(i * 200):((i + 1) * 200)])

            zc2[i]=np.mean(zm2[(i * 200):((i + 1) * 200)])
            uc2[i] = np.mean(um2[(i * 200):((i + 1) * 200)])
            vc2[i] = np.mean(vm2[(i * 200):((i + 1) * 200)])

        autoMedie = ((m_v + m_veh)/2).astype(int)
        m_v = m_v.astype(int)
        m_veh = m_veh.astype(int)
        #autoMedie = np.concatenate((m_v,m_veh),axis=0)
        print(autoMedie)
        print(m_veh)
        plt.errorbar(m_veh,zc2,yerr=std_error,marker="x",color="orange",markersize=2,label="training indipendente")

        print("valor medio di total stopped :", zc2.mean())
        print("massimo di total stopped :", zc2.max(),"su max veicoli : ", m_veh.max())
        tick=np.arange(0,1000,100)
        plt.xticks(autoMedie, autoMedie)
        plt.legend()
        plt.grid()
        plt.show()

        #mean waiting time

        std_error = np.std(um1, ddof=1) / np.sqrt(len(um1))
        # create chart
        plt.errorbar(m_v,uc1,yerr=std_error,marker="x",color="blue", label="training copiato")
        print("valor medio di mean waiting time :" ,uc1.mean())


        print("massimo di mean waiting time :", uc1.max())
        std_error = np.std(um2, ddof=1) / np.sqrt(len(um2))
        # create chart
        plt.errorbar(m_veh,uc2,yerr=std_error,marker="o",color="orange",markersize=2,label="training indipendente")
        print("valor medio di mean waiting time :" ,uc2.mean())


        print("massimo di mean waiting time :", uc2.max())
        plt.xlabel("numero auto medio in circolazione ogni 1000 secondi",size=12)
        plt.title("rete 4x4 traffico crescente reward wait misura mean waiting time ")
        plt.ylabel("system mean waiting time(seconds)",size=12)

        plt.legend()
        plt.xticks(autoMedie,autoMedie)
        plt.grid()
        plt.show()
        #plt.plot(x, vm)

        #velocità media





        std_error = np.std(vm1, ddof=1) / np.sqrt(len(vm1))
        # create chart
        plt.errorbar(m_v, vc1, yerr=std_error, marker="o", color="blue", markersize=2,label="training copiato")
        std_error = np.std(vm2, ddof=1) / np.sqrt(len(vm2))
        plt.errorbar(m_veh, vc2, yerr=std_error, marker="x", color="orange",markersize=2, label="training indipendente")
        plt.xlabel("numero auto medio in circolazione ogni 1000 secondi",size=12)
        plt.title("4x4 traffico crescente reward wait misura mean speed")
        plt.legend()
        plt.xticks(autoMedie, autoMedie)
        plt.grid()
        plt.ylabel("system mean speed(Km/h)",size=12)
        plt.show()
if __name__ == '__main__':
    file2='D:/programmi/sumo/esperimenti semafori/Reinforcement-learning-in-traffic-light/outputs/4x4/ql-4x4grid-variReward-10000sec(e auto)'
    plotResult('D:/programmi/sumo/esperimenti semafori/Reinforcement-learning-in-traffic-light/outputs/4x4/ql-4x4grid-unareward',file2)