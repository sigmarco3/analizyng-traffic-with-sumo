import argparse
import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Callable, Optional, Tuple, Union
import sumolib
from datetime import datetime
from env import SumoEnvironment
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import traci

import plotResult_single as pl

env = None
delta_time = 5
metrics =[]



# def get_per_agent_info():
#     stopped = [env.traffic_signals[ts].get_total_queued() for ts in env.ts_ids]
#     accumulated_waiting_time = [sum(env.traffic_signals[ts].get_accumulated_waiting_time_per_lane()) for ts in
#                                 env.ts_ids]
#     average_speed = [env.traffic_signals[ts].get_average_speed() for ts in env.ts_ids]
#     info = {}
#     for i, ts in enumerate(env.ts_ids):
#         info[f'{ts}_stopped'] = stopped[i]
#         info[f'{ts}_accumulated_waiting_time'] = accumulated_waiting_time[i]
#         info[f'{ts}_average_speed'] = average_speed[i]
#     info['agents_total_stopped'] = sum(stopped)
#     info['agents_total_accumulated_waiting_time'] = sum(accumulated_waiting_time)
#     return info
def get_system_info():
    vehicles = traci.vehicle.getIDList()
    speeds = [traci.vehicle.getSpeed(vehicle) for vehicle in vehicles]
    waiting_times = [traci.vehicle.getWaitingTime(vehicle) for vehicle in vehicles]
    totals = traci.vehicle.getIDCount()
    return {
        # In SUMO, a vehicle is considered halting if its speed is below 0.1 m/s
        'system_total_stopped': sum(int(speed < 0.1) for speed in speeds),
        'system_total_waiting_time': sum(waiting_times),
        'system_mean_waiting_time': np.mean(waiting_times),
        'system_mean_speed': 0.0 if len(vehicles) == 0 else np.mean(speeds),
        'total_vehicle': totals
    }
def computeInfo():
    info = {'step': traci.simulation.getTime()}

    info.update(get_system_info())

    metrics.append(info)
    return info

def run(seconds,run,programs):
    index=0
    while traci.simulation.getTime()<seconds:
        for ts in traci.trafficlight.getIDList():
            index = index % 8
            traci.trafficlight.setPhase(ts,index)
            endPhase= traci.trafficlight.getNextSwitch(ts)
        while(traci.simulation.getTime()<endPhase):
            if traci.simulation.getTime()%5==0:
                info = computeInfo()
            traci.simulationStep()
        index = index + 1

    traci.close()

def save_csv( out_csv_name, run):
        if out_csv_name is not None:
            df = pd.DataFrame(metrics)
            Path(Path(out_csv_name).parent).mkdir(parents=True, exist_ok=True)
            df.to_csv(out_csv_name + '_conn0_run{}'.format( run) + '.csv', index=False)
if __name__ == '__main__':

    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  description="""Q-Learning Single-Intersection""")
    prs.add_argument("-route", dest="route", type=str, default='nets/2way-single-intersection/single-random4.rou.xml', help="Route definition xml file.\n")
    prs.add_argument("-a", dest="alpha", type=float, default=0.1, required=False, help="Alpha learning rate.\n")
    prs.add_argument("-g", dest="gamma", type=float, default=0.99, required=False, help="Gamma discount rate.\n")
    prs.add_argument("-e", dest="epsilon", type=float, default=0.05, required=False, help="Epsilon.\n")
    prs.add_argument("-me", dest="min_epsilon", type=float, default=0.005, required=False, help="Minimum epsilon.\n")
    prs.add_argument("-d", dest="decay", type=float, default=0.997, required=False, help="Epsilon decay.\n")
    prs.add_argument("-mingreen", dest="min_green", type=int, default=10, required=False, help="Minimum green time.\n")
    prs.add_argument("-maxgreen", dest="max_green", type=int, default=30, required=False, help="Maximum green time.\n")
    prs.add_argument("-gui", action="store_true", default=False, help="Run with visualization on SUMO.\n")
    prs.add_argument("-fixed", action="store_true", default=False, help="Run with fixed timing traffic signals.\n")
    prs.add_argument("-s", dest="seconds", type=int, default=10000, required=False, help="Number of simulation seconds.\n")
    prs.add_argument("-r", dest="reward", type=str, default='diff-waiting-time', required=False, help="Reward function: [-r queue] for average queue reward or [-r wait] for waiting time reward.\n")
    prs.add_argument("-runs", dest="runs", type=int, default=1, help="Number of runs.\n")
    args = prs.parse_args()
    experiment_time = str(datetime.now()).split('.')[0]
    out_csv = 'outputs/2x2/result-2x2 crescente-static'
    output_file = 'output-pressure2(curriculum).csv'
    net = "nets/2way-single-intersection/single-intersection.net.xml"
    if args.gui :
        sumo_binary = sumolib.checkBinary('sumo-gui')
    else:
        sumo_binary = sumolib.checkBinary('sumo')


    sumo_cmd = [sumo_binary,
                '-n', net,
                '-r', args.route]
    sumo_cmd.extend(['--start', '--quit-on-end'])
    for runs in range(1, args.runs+1):
        metrics = []
        #inStates = env.reset()
        traci.start(sumo_cmd)
        if args.gui :
            traci.gui.setSchema(traci.gui.DEFAULT_VIEW, "real world")
        for ts in traci.trafficlight.getIDList():
            programs = traci.trafficlight.getAllProgramLogics(ts)
            logic = programs[0]
            traci.trafficlight.setProgramLogic(ts, logic)
        run(args.seconds,runs,programs) #da sistemare

        if runs!=0:
          save_csv(out_csv, runs)
    #     env.close()


















