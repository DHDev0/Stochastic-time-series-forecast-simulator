#defaul python lib
import requests
import ast
import time
from time import gmtime
from datetime import datetime, timedelta
import sqlite3
import pickle

#lib to add
import pandas as panda
import gym
from gym import spaces
import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

class forecast_crypto(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, render_mode=None,symb='BTCUSDT', mode = "training"):
            
        self.render_mode = render_mode
        self.window_size = 24  # The size of the Game window
        self.computed_indice = 6
        self.store_reconstruct_predict = None
        self.render_origine = 0
        self.max_action_space_size = 100
        self.symb = symb
        self.database_name = self.symb.lower()
        self.dataset_name = "historical_data"
        self.table_content = ("open_time", "open", "high", "low", "close", "volume", "close_time", "qav", "num_trades", "taker_base_vol", "taker_quote_vol", "ignore")
        interval = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h","1d", "3d", "1w", "1M"]
        self.interval = interval[5]
        dataset = self.generate_dataset(database_name = self.database_name,
                                        dataset_name = self.dataset_name,
                                        table_content = self.table_content, 
                                        interval = self.interval,
                                        symbole = self.symb)
        
        self.mode = mode
        
        #dataset structure
        # ['open_time','open', 'high', 'low', 'close', 'volume','close_time', 'qav','num_trades','taker_base_vol','taker_quote_vol', 'ignore']
        # obs strcture
        #obs ['month','day','hour','open', 'high', 'low', 'close', 'volume', 'qav','num_trades','taker_base_vol','taker_quote_vol']
        
        obs, targ = None , None
        for i,ii in zip(dataset[1:],dataset[:-1]):
            if obs is None:
                obs , targ = self.concatenation(i,ii)
            else:
                update_obs , update_targ = self.concatenation(i,ii)
                obs = np.vstack((obs, update_obs))
                targ = np.vstack((targ, update_targ))

        self.bound = self.find_bound(obs[:,self.computed_indice],max_action_space_size = self.max_action_space_size)
        if self.mode == "training":
            self.dataset_obs = obs[:len(obs)*0.95]
            self.dataset_targ = targ[:len(obs)*0.95]
        if self.mode == "validation":
            self.dataset_obs = obs[len(obs)*0.95:]
            self.dataset_targ = targ[len(obs)*0.95:]

        obs_array = np.array([np.finfo(np.float32).max for i in range(len(obs[0]))])
        self.observation_space = spaces.Box(-obs_array, obs_array, dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.bound))
        self._action_to_direction = dict(zip(list(range(len(self.bound))),self.bound))

    def interval_generator(self,interval):
        value, unit = int(interval[:-1]), interval[-1] #check make global
        if unit == 'm': return timedelta(minutes=value)
        elif unit == 'h': return timedelta(hours=value)
        elif unit == 'd': return timedelta(days=value)
        elif unit == 'w': return timedelta(weeks=value)
        elif unit == 'M': return timedelta(days=30*value)
        
    def update_database(self,conn,dataset,table_content,dataset_name):
        n_value_of_table_content = tuple(["?" for _ in range(len(table_content))])
        tc_to_str = str(table_content).replace("'", "")
        nv_to_str = str(n_value_of_table_content).replace("'", "")
        conn.execute(f'''CREATE TABLE IF NOT EXISTS {dataset_name} {tc_to_str}''')
        for values in dataset:
            conn.execute(f"INSERT INTO {dataset_name} {tc_to_str} VALUES {nv_to_str}", values)
        conn.commit()
        
    def retrieve_data_from_dataset(self,conn,dataset_name : str = "btcusdt"):
        cursor = conn.execute(f"SELECT * FROM {dataset_name}")
        return [list(row) for row in cursor]

    def min_distance_time(self,recent_row,interval):
        index_last_time = 6 #check make global
        last_time = str(gmtime(int(recent_row[index_last_time])/1000)) 
        if isinstance(recent_row[index_last_time],(int,float)) :
            last_time = time.strftime('%Y-%m-%d %H:%M:%S', gmtime(int(recent_row[index_last_time])/1000))
        else: last_time = recent_row[index_last_time]
        last_time = datetime.strptime(last_time, '%Y-%m-%d %H:%M:%S')
        time_now = datetime.utcnow()
        interval_dist = self.interval_generator(interval)
        return time_now - last_time > interval_dist*2 #check make global   minimum batch betfore update
        
    def connect_to_database(self,database_name : str = "btcusdt"):
        return sqlite3.connect(f'{database_name}.db')
        
    def disconnect_to_database(self,conn):
        conn.close()

    def query_data(self,symb,interval,init_time,end_time):
        response = requests.get(f'https://api.binance.com/api/v3/klines?symbol={symb}&interval={interval}&startTime={init_time}&endTime={end_time}&limit=1000')
        return ast.literal_eval(response.__dict__['_content'].decode('UTF-8'))
    
    def klinedata(self,last_date = None,
                  interval : str = "1h",
                  symbole : str = "BTCUSDT"):
        overall=[]
        end_date = datetime.utcnow()
        batch_size = self.interval_generator(interval)*960 #960 is the max api request size

        if last_date is None:
            backward_date = timedelta(days=720,hours=1, minutes=0,seconds=0, microseconds=0) # maximum past kline to request
            start_date = end_date - backward_date
        else:#provide close_time here
            convert_start = time.strftime('%Y-%m-%d %H:%M:%S', gmtime(int(last_date)/1000))
            start_date = datetime.strptime(convert_start, '%Y-%m-%d %H:%M:%S')

        total_time = end_date - start_date
        number_of_batch = total_time/batch_size
        integer_list = list(range(int(number_of_batch) + 1)) + [number_of_batch]

        for i in range(len(integer_list)-1):
            init_time = start_date + (batch_size * integer_list[i])
            edge_time = start_date + (batch_size * integer_list[i+1])
            converted_init = int(init_time.timestamp() * 1000)
            converted_edge = int(edge_time.timestamp() * 1000)
            res = self.query_data(symbole,interval,converted_init,converted_edge)
            if not isinstance(res,dict): overall+=res
            print(f"Update dataset -> batch number : {i} on {integer_list[-2]} ")
            min_laps , max_laps = 2 , 50
            time.sleep(np.random.uniform(min_laps, max_laps))
            
        res_rf = [[float(h) for h in i] for i in overall]
        if not res_rf: raise Exception("empty output")
        return res_rf
    
    def check_last_row(self,conn,dataset_name,table_content):
        cursor = conn.cursor()
        query = f"SELECT * FROM {dataset_name} ORDER BY {table_content[0]} DESC LIMIT 1"
        try:
            cursor.execute(query)
            recent_row = cursor.fetchone()
        except:
            recent_row = None
        return recent_row
    
    def generate_dataset(self,                         
                         database_name : str = "btcusdt",
                         dataset_name : str = "kline_history",
                         table_content : tuple = ("open_time", "open"), 
                         interval : str = "1h",
                         symbole : str = "BTCUSDT"):
        conn = self.connect_to_database(database_name)
        recent_row = self.check_last_row(conn,
                                         dataset_name,
                                         table_content)
        
        if recent_row is None:
            base_data = self.klinedata(last_date = None, 
                                       interval = interval,
                                       symbole = symbole)
            self.update_database(base_data)
            del base_data
            
        elif self.min_distance_time(recent_row, interval):
            last_time = recent_row[6] #last close time #check make global
            updated_data = self.klinedata(last_date = last_time, 
                                          interval = interval,
                                          symbole = symbole)
            self.update_database(conn, 
                                 updated_data, 
                                 table_content, 
                                 dataset_name)
            del updated_data

        dataset = self.retrieve_data_from_dataset(conn,
                                                  dataset_name)
        self.disconnect_to_database(conn)
        return dataset

    def unix_to_time(self,x):
        return list(gmtime(int(x)/1000)[1:-5])

    def converter(self,x):
        converted = np.array(x[1:6] + x[7:-1]) #check make global
        # timestamp = np.array(unix_to_time(x[0]) + unix_to_time(x[6]))
        index_to_take_time = 6 #check make global
        timestamp = np.array(self.unix_to_time(x[index_to_take_time]))
        where_0 = np.where(converted <= 0)
        converted[where_0] = 1
        return (converted,timestamp)

    def concatenation(self,x,y):
        conversion_next = self.converter(x)
        conversion_init = self.converter(y)
        combine = conversion_next[0]/conversion_init[0]
        observation = np.concatenate((conversion_next[1], combine), axis=-1)
        target = np.concatenate((conversion_next[1], conversion_next[0]), axis=-1)
        return observation , target

    def find_bound(self,x, max_action_space_size = 100):
        # sourcery skip: raise-specific-error

        for i in list(range(1,10))[::-1]:
            bound = np.unique(np.round(np.sort(x), i)).tolist()
            if len(bound) <= max_action_space_size:
                s = panda.Series(bound)
                print("action space size: ",len(bound))
                print("actual action space: ",bound)
                s.plot()
                return bound
        raise Exception(f"can't find {max_action_space_size} or less bound of action")
    
##################################### gym mechanism ##################################################
     
    def _get_obs(self):
            self.observation = self.dataset_obs[self.index + self.origin]
            return self.observation.astype(np.float32)
        
            
    def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.index = int(self.np_random.integers(0, len(self.dataset_obs)-self.window_size, size=1, dtype=np.int64))
            self.origin = 0
            observation = self._get_obs()
            return observation, {}    
        
        
    def step(self, action):
            act = self._action_to_direction[action]
            if self.origin == 0:
                self.predict_value = np.array([self.dataset_targ[self.index + self.origin,self.computed_indice] * act])
            else:
                self.predict_value = np.concatenate((self.predict_value, self.predict_value[-1] * act), axis=None)
            self.actual_value = self.dataset_targ[self.index + 1 : self.index + self.origin + 2,self.computed_indice]
            
            difference = (self.predict_value/self.actual_value).mean()
            self.origin += 1 
            
            reward = int((100 - (abs(1 - difference)*100))/10)#np.absolute(self.bound-x).argmin()  np.where(arr == 15)
            terminated = self.origin == self.window_size or not (0.95 < difference < 1.05 )
            observation = self._get_obs()
            info = {}

            return observation, reward, terminated, False, info
        

    def init_render(self):
        self.root = tk.Tk()
        self.count = 0
        x, y = [1, 2, 3] , [0, 0, 0]
        self.figure,self.ax = plt.subplots()
        self.ax.plot(x, y)
        self.canvas = FigureCanvasTkAgg(self.figure, self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.draw()
        #keep windows in front
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.attributes('-topmost', False)
        
        self.root.update()
        
    def update_render(self, *time_serie):
        #take unlimited number of 1d array
        self.ax.clear()
        hold_range_max , hold_data_max = 0 , 0
        hold_range_min , hold_data_min = 0 , 0
        self.ax = self.canvas.figure.axes[0]
        for i in time_serie:
            range_v = np.arange(0, len(i), dtype=int)
            data_v = i
            self.ax.plot(range_v, data_v)
            hold_range_max = hold_range_max if max(range_v) < hold_range_max else max(range_v)
            hold_data_max  = hold_data_max if max(data_v) < hold_data_max else max(data_v)
            hold_range_min = hold_range_min if min(range_v) > hold_range_min else min(range_v)
            hold_data_min  = hold_data_min if min(data_v) > hold_data_min else min(data_v)
        self.ax.xaxis.set_label_text("iteration")
        self.ax.yaxis.set_label_text("price")
        # self.ax.set_xlim(hold_range_min, hold_range_max)
        # self.ax.set_ylim(hold_data_min, hold_data_max)   
        self.canvas.draw()
        self.root.update()

        
    def close_render(self):
        self.root.destroy()

    def render(self):
        #call render after step
        if self.render_mode == "human":
            if self.render_origine == 0:
                self.init_render()
                self.render_origine += 1 
            self.update_render(self.predict_value,self.actual_value)


    def close(self):
        if self.render_mode == "human":
            self.close_render()
        del self.dataset_obs
        del self.dataset_targ



##### test code
# env = forecast_crypto(render_mode="human",symb='BTCUSDT')
# env.reset()
# for i in range(30):
#     action = np.random.choice(list(env._action_to_direction.keys()))
#     env.step(action)
#     env.render()
#     time.sleep(2)
# env.close()



