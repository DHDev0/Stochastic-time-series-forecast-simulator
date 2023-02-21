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
    
    def __init__(self, 
                 render_mode=None,
                 symb=['ETHUSDT','BTCUSDT','DOGEUSDT'], 
                 mode = "training",
                 future_window = 5,
                 number_of_past_observation = 10,
                 computed_asset = [ 0 , 1 , 2],
                 computed_value = [ 1 , 2 , 3 , 4]):
            
        self.render_mode = render_mode
        
        self.future_window = future_window  # The number of step you want to predict ahead
        self.number_of_past_observation = number_of_past_observation
        
        self.store_reconstruct_predict = None
        self.render_origine = 0
        
        self.symb = [symb] if isinstance(symb,str) else symb
        self.dataset_name = "historical_data"
        self.table_content = ("open_time", "open", "high", "low", "close", "volume", "close_time", "qav", "num_trades", "taker_base_vol", "taker_quote_vol", "ignore")
        interval = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h","1d", "3d", "1w", "1M"]
        self.interval = interval[5]
        self.database_name = [ i.lower()+"_"+self.interval for i in self.symb]
        
        self.observation_indice = [ i for i in range(9)]
        
        self.computed_value = [ 1 , 2 , 3 , 4]
        self.computed_asset = computed_asset
        self.max_action_space_size = 700
        
        self.mode = mode
        
        #init db class for upload database to transform db
        db = Database_build()
        
        #refresh dataset untill fully refresh
        db.refresh_db(self.symb,
                       self.database_name,
                       self.dataset_name,
                       self.table_content,
                       self.interval)
        #loop/stack specified dataset to batch
        full_target , full_observation = None , None
        for symbole,symbole_database_name in zip(self.symb,self.database_name):
            #dataset structure
            #indice: 0       1       2       3       4         5          6         7         8             9                 10            11
            # ['open_time','open', 'high', 'low', 'close', 'volume','close_time', 'qav','num_trades','taker_base_vol','taker_quote_vol', 'ignore']
            dataset = db.generate_dataset(database_name = symbole_database_name,
                                            dataset_name = self.dataset_name,
                                            table_content = self.table_content, 
                                            interval = self.interval,
                                            symbole = symbole)
            # obs strcture
            #indice:         0           1       2       3       4         5       6        7               8                9 
            #obs ['month_day_hour_min','open', 'high', 'low', 'close', 'volume', 'qav','num_trades','taker_base_vol','taker_quote_vol']       
            obs, targ = db.transform_to_kline_support(dataset)
            #use indice: target is the actual price history, and obs is transformed dataset (add dim for batch stacking)
            obs, targ = obs[:,np.newaxis,self.observation_indice], targ[:,np.newaxis,self.observation_indice]
            #pair stacking to batch
            full_observation, full_target = db.data_stacking(obs, targ, full_observation, full_target)
        
        #select dataset size
        self.sliced = self.slice_mode(full_observation)
        self.dataset_obs = full_observation[self.sliced]
        self.dataset_targ = full_target[self.sliced]
        assert self.dataset_obs.shape[0] >= max(self.number_of_past_observation*8,self.number_of_past_observation+100) , f"minimum number of stack past observation{max(self.number_of_past_observation*8,self.number_of_past_observation+100)}"

        

        #find action space of indice you want to compute/predict
        # bound = []
        # for i in self.computed_value:
        #     bound += db.find_bound(self.dataset_obs[...,self.computed_value],max_action_space_size = self.max_action_space_size)
        bound = [0.64, 0.709, 0.738, 0.746, 0.747, 0.78, 0.802, 0.811, 0.82, 0.823, 0.828, 0.831, 0.844, 0.847, 0.848, 0.849, 0.85, 0.851, 0.854, 0.858, 0.863, 0.867, 0.868, 0.869, 0.87, 0.873, 0.875, 0.876, 0.88, 0.881, 0.882, 0.883, 0.884, 0.886, 0.887, 0.888, 0.889, 0.89, 0.893, 0.894, 0.895, 0.897, 0.898, 0.899, 0.9, 0.901, 0.902, 0.903, 0.904, 0.905, 0.906, 0.907, 0.908, 0.909, 0.91, 0.911, 0.912, 0.913, 0.914, 0.915, 0.916, 0.917, 0.918, 0.919, 0.92, 0.921, 0.922, 0.923, 0.924, 0.925, 0.926, 0.927, 0.928, 0.929, 0.93, 0.931, 0.932, 0.933, 0.934, 0.935, 0.936, 0.937, 0.938, 0.939, 0.94, 0.941, 0.942, 0.943, 0.944, 0.945, 0.946, 0.947, 0.948, 0.949, 0.95, 0.951, 0.952, 0.953, 0.954, 0.955, 0.956, 0.957, 0.958, 0.959, 0.96, 0.961, 0.962, 0.963, 0.964, 0.965, 0.966, 0.967, 0.968, 0.969, 0.97, 0.971, 0.972, 0.973, 0.974, 0.975, 0.976, 0.977, 0.978, 0.979, 0.98, 0.981, 0.982, 0.983, 0.984, 0.985, 0.986, 0.987, 0.988, 0.989, 0.99, 0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998, 1.002, 1.003, 1.004, 1.005, 1.006, 1.007, 1.008, 1.009, 1.01, 1.011, 1.012, 1.013, 1.014, 1.015, 1.016, 1.017, 1.018, 1.019, 1.02, 1.021, 1.022, 1.023, 1.024, 1.025, 1.026, 1.027, 1.028, 1.029, 1.03, 1.031, 1.032, 1.033, 1.034, 1.035, 1.036, 1.037, 1.038, 1.039, 1.04, 1.041, 1.042, 1.043, 1.044, 1.045, 1.046, 1.047, 1.048, 1.049, 1.05, 1.051, 1.052, 1.053, 1.054, 1.055, 1.056, 1.057, 1.058, 1.059, 1.06, 1.061, 1.062, 1.063, 1.064, 1.065, 1.066, 1.067, 1.068, 1.069, 1.07, 1.071, 1.072, 1.073, 1.074, 1.075, 1.076, 1.077, 1.078, 1.079, 1.08, 1.081, 1.082, 1.083, 1.084, 1.085, 1.086, 1.087, 1.088, 1.089, 1.09, 1.091, 1.092, 1.093, 1.094, 1.095, 1.096, 1.097, 1.098, 1.099, 1.1, 1.101, 1.102, 1.103, 1.105, 1.107, 1.108, 1.109, 1.11, 1.111, 1.112, 1.113, 1.114, 1.115, 1.116, 1.117, 1.118, 1.119, 1.12, 1.121, 1.122, 1.123, 1.124, 1.125, 1.126, 1.129, 1.13, 1.131, 1.133, 1.135, 1.136, 1.137, 1.138, 1.139, 1.14, 1.141, 1.144, 1.145, 1.149, 1.15, 1.151, 1.152, 1.154, 1.155, 1.157, 1.159, 1.161, 1.163, 1.164, 1.166, 1.168, 1.17, 1.178, 1.18, 1.183, 1.185, 1.195, 1.204, 1.205, 1.21, 1.212, 1.228, 1.233, 1.234, 1.24, 1.242, 1.249, 1.252, 1.261, 1.308, 1.33]
        self.bound = sorted(list(set(bound)))
        print("action space size: ",len(self.bound))
        print("actual action space: ",self.bound)   
        
        #TO FIX WITH CHOICE OF DATA AUGMENTATION
        obs_sh = list(full_observation[:self.number_of_past_observation,...].shape)
        obs_sh[1] = int(obs_sh[1]*6)
        obs_sh = tuple(obs_sh)
        
        obs_array = np.full(obs_sh, np.finfo(np.float32).max) #fix [0] for the size of the batch observation
        self.observation_space = spaces.Box(-obs_array, obs_array, dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.bound))
        self._action_to_direction = dict(zip(list(range(len(self.bound))),self.bound))


    def slice_mode(self,full_observation):
        if self.mode == "training":
            training_slice = -(len(full_observation) - int(len(full_observation)*0.10))          
            slicer = slice( None , -training_slice )
        if self.mode == "validation":
            back_validation_slice = len(full_observation) - int(len(full_observation)*0.1)
            forw_validation_slice = len(full_observation) - int(len(full_observation)*0.05)
            slicer = slice( -back_validation_slice , -forw_validation_slice  )
        if self.mode == "live":
            live_slice = -(len(full_observation) - int(len(full_observation)*0.05))
            slicer = slice( -live_slice  , None )
        if self.mode == "overall":
            slicer = slice( 0  , len(full_observation) )
        return slicer
    
    def reward_function(self,target : np.array,prediction: np.array):
        # reward = 1 - ( np.abs(prediction - target)/ ((np.abs(target)+np.abs(prediction))/2) )#SMAPE
        # return int(np.exp(reward * np.log(100)))
        difference = prediction/target    
        difference = max(min(difference,1.99),0.01)
        reward = np.round((1 - abs(1 - difference))*100,1)
        return reward
##################################### gym mechanism ##################################################        
            
    def reset(self, seed=None, options=None):#to fix
            super().reset(seed=seed)
            
            self.initiate = True
            
            self.origin = -1
            
            if self.mode == "live":
                self.index = len(self.dataset_obs)
            else:
                
                min_bound = max(self.number_of_past_observation*8,self.number_of_past_observation+100)
                self.index = int(self.np_random.integers(min_bound, len(self.dataset_obs)-self.future_window , size=1, dtype=np.int64))
            
            self.walk = walker(self.computed_asset,self.computed_value)
            
            #observation
            self.simulation_observation = self.dataset_obs[self.index - self.number_of_past_observation : self.index ]            
            fd_obs = indicator.function_to_cumulate(indicator.fractal_dimension , self.dataset_obs[self.index - self.number_of_past_observation - 100 : self.index ])
            ae_obs = indicator.function_to_cumulate(indicator.approximate_entropy , self.dataset_obs[self.index - self.number_of_past_observation - 100 : self.index ])
            mn2_obs = indicator.keep_every_nth_value(2, self.dataset_obs[self.index - (self.number_of_past_observation*2) : self.index ])
            mn4_obs = indicator.keep_every_nth_value(4, self.dataset_obs[self.index - (self.number_of_past_observation*4) : self.index ])
            mn8_obs = indicator.keep_every_nth_value(8, self.dataset_obs[self.index - (self.number_of_past_observation*8) : self.index ])
            self.simulation_observation = np.concatenate((self.simulation_observation, fd_obs, ae_obs ,mn2_obs,mn4_obs,mn8_obs ), axis=1)
            #target
            self.simulation_observation_without_transform = self.dataset_targ[self.index - self.number_of_past_observation : self.index ]

            return self.simulation_observation.astype(np.float32), {}    
        
        
    def step(self, action):#to fix
        #cyclic target
        self.crypto_index,self.value_index,initiate = self.walk.step()
        
        # observation set up
        if self.initiate:
            self.origin += 1 
            self.simulation_observation = np.append(self.simulation_observation,np.full(self.simulation_observation[:1].shape,0),axis=0)
            self.simulation_observation_without_transform= np.append(self.simulation_observation_without_transform,np.full(self.simulation_observation_without_transform[:1].shape,0),axis=0)

        #action
        act = self._action_to_direction[action]
        
        #observation
        self.simulation_observation[-1 , self.crypto_index , self.value_index] =  act
        observation = self.simulation_observation[self.origin +1 :  ].astype(np.float32)
        self.simulation_observation_without_transform[-1 , self.crypto_index , self.value_index] =  self.simulation_observation_without_transform[-2 , self.crypto_index , self.value_index]*act

        # reward
        if self.mode == "live":
            reward = 0
        else:
            self.predict_value = self.simulation_observation_without_transform[-1 , self.crypto_index , self.value_index]
            self.actual_value =  self.dataset_targ[self.index + self.origin + 1 , self.crypto_index , self.value_index]           
            reward = self.reward_function(self.actual_value,self.predict_value)    

        #game state(terminate or continue)
        terminated = self.origin == self.future_window - 1
        
        #cyclic parameter update
        self.initiate = initiate

        info = {}

        return observation, reward, terminated, info
        

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
        
    def update_render(self, *time_serie, mode = "table"):
        #take unlimited number of 1d array
        self.ax.clear()
        if mode == "plot":
        # # create plot
            self.ax = self.canvas.figure.axes[0]
            for data_v in time_serie:
                range_v = np.tile(np.arange(data_v.shape[-1], dtype=int), data_v.shape[:-1]+(1,))    
                self.ax.plot(range_v, data_v)
            self.ax.xaxis.set_label_text("iteration")
            self.ax.yaxis.set_label_text("price") 
    
        if mode == "table":
        # # create table
            self.ax.axis('off')
            self.ax.axis('tight')
            headers = [f"t_{i}" for i in range(len(time_serie[0]))]
            # rows = [0] + [ "target " for i in range(len(time_serie) // 2)] + [ "prediction " for i in range(len(time_serie) // 2)]
            table_data = [headers] + [i for i in time_serie]
            the_table = self.ax.table(cellText=table_data, loc='center')#, rowLabels=rows,)
            the_table.auto_set_font_size(False)
            the_table.set_fontsize(8)
            the_table.scale(1.5, 1.5)
            self.figure.savefig('fox.png')
        
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
            prediction = self.simulation_observation_without_transform[self.origin:,:,self.computed_value].transpose(1,2,0)
            prediction = tuple(np.round(i,2) for i in prediction.reshape(prediction.shape[0]*prediction.shape[1],prediction.shape[2]))
            overall = prediction
            if self.mode == "live":
                pass
            else:
                target = self.dataset_targ[self.index - self.number_of_past_observation + self.origin: self.index + self.origin + 1,:,self.computed_value ].transpose(1,2,0)
                target = tuple(np.round(i,2) for i in target.reshape(target.shape[0]*target.shape[1],target.shape[2]))
                overall = target+prediction
            self.update_render(*overall)


    def close(self):
        if self.render_mode == "human":
            self.close_render()
        self.walk.reset()
        del self.dataset_obs
        del self.dataset_targ

class walker:
    def __init__(self,symbole_set , indice_set):
        self.sy_set = symbole_set
        self.ind_set = indice_set
        self.symbole_set = len(symbole_set)
        self.indice_set = len(indice_set)
        self.symbole_set_origin = -1
        self.indice_set_origin = -1
    def step(self):
        self.indice_set_origin +=1
        if self.indice_set_origin % self.indice_set == 0 :
            self.symbole_set_origin += 1
        self.indice_set_origin = self.indice_set_origin % self.indice_set
        self.symbole_set_origin = self.symbole_set_origin % self.symbole_set
        actual_step = (self.sy_set[self.symbole_set_origin],self.ind_set[self.indice_set_origin])
        full_cycle = [actual_step[0],actual_step[1]] == [self.sy_set[-1],self.ind_set[-1]]
        return (actual_step[0],actual_step[1],full_cycle)
    def reset(self):
        self.symbole_set_origin = -1
        self.indice_set_origin = -1


class Database_build:
    def __init__(self) -> None:
        pass
    
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
        return time_now - last_time > interval_dist*1 #check make global   minimum batch betfore update
        
    def connect_to_database(self,database_name : str = "btcusdt"):
        # return sqlite3.connect(f'C:\\Users\\ddery\\anaconda3\\envs\\pentos\\Lib\\site-packages\\gym\\envs\\forecast\\{database_name}.db')
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
            if not isinstance(res,dict): 
                overall+=res
                print(f"Update dataset {symbole} -> batch number : {i} on {integer_list[-2]} ")
            min_laps , max_laps = 15 , 20
            time.sleep(np.random.uniform(min_laps, max_laps) if len(integer_list)-1 > 0 else 0)#4 request max per min for binance api
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
            self.update_database(conn, 
                                 base_data, 
                                 table_content, 
                                 dataset_name)
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
        self.fill_missing_timestamps(conn, dataset_name, interval)
        dataset = self.retrieve_data_from_dataset(conn,
                                                  dataset_name)
        self.disconnect_to_database(conn)
        return dataset

    def unix_to_time(self,x):
        #time_of_month_day_hour_min 
        tomdhm = gmtime(int(x)/1000)[1:-4]
        return [((tomdhm[0]/12) + (tomdhm[1]/30) + (tomdhm[2]/24) + (tomdhm[3]/60))/4]

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
                return bound
        raise Exception(f"can't find {max_action_space_size} or less bound of action")
    
    def transform_to_kline_support(self,dataset):
        obs, targ = None , None
        for i,ii in zip(dataset[1:],dataset[:-1]):
            if obs is None:
                obs , targ = self.concatenation(i,ii)
            else:
                update_obs , update_targ = self.concatenation(i,ii)
                obs = np.vstack((obs, update_obs))
                targ = np.vstack((targ, update_targ))
        return obs , targ 

    def fill_missing_timestamps(self,conn, table_name, time_interval):
        df = panda.read_sql_query(f"SELECT * FROM {table_name}", conn)
        df['open_time'] = panda.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = panda.to_datetime(df['close_time'], unit='ms')
        df.set_index('open_time', inplace=True)
        start = df.index[0]
        end = df.index[-1]
        new_index = panda.date_range(start, end, freq=time_interval)
        try:
            df = df.reindex(new_index, method='nearest',copy=False)
            time_interval_timedelta = panda.Timedelta(time_interval)
            df['close_time'] = df.index + time_interval_timedelta
            df['open_time'] = (df.index.astype(np.int64) / 10**9).astype(np.int64) * 10**3
            df['close_time'] = ((df['close_time'].astype(np.int64) / 10**9).astype(np.int64) * 10**3)-1
            df.insert(0, 'open_time', df.pop('open_time'))
            df.drop_duplicates( keep="first")
            df.to_sql(table_name, conn, if_exists='replace',index=False)
        except:
            pass
        
    def data_stacking(self,obs, targ, full_observation, full_target):
        if full_observation is None or full_target is None :
            full_observation  , full_target = obs, targ
            loop_count = False
        else: # need fix
            obs_len , targ_len = min(full_observation.shape[0], obs.shape[0]) , min(full_target.shape[0], targ.shape[0])                
            full_observation = np.append(full_observation[full_observation.shape[0]-obs_len:], obs[obs.shape[0]-obs_len:], axis=1)
            full_target = np.append(full_target[full_target.shape[0]-targ_len:], targ[targ.shape[0]-targ_len:], axis=1)
        return full_observation,full_target
        
    def refresh_db(self,
                    symb,
                    database_name,
                    dataset_name,
                    table_content,
                    interval,
                    ):
        dataset_refresh_needed = True
        while dataset_refresh_needed:
            start_time = datetime.utcnow()
            for symbole,symbole_database_name in zip(symb,database_name):
                dataset = self.generate_dataset(database_name = symbole_database_name,
                                                dataset_name = dataset_name,
                                                table_content = table_content, 
                                                interval = interval,
                                                symbole = symbole)
            time_difference = datetime.utcnow() - start_time
            dataset_refresh_needed = not time_difference <= self.interval_generator(interval)
      
            
class indicator:
    
    @staticmethod
    def function_to_cumulate(function , batch):
        batch_shape = batch.shape 
        # batch_shape[0] -> number of timestamp
        # batch_shape[1] -> number of assets
        # batch_shape[2] -> number of values
        new_assets = np.zeros((batch_shape[0] - 100, 1, batch_shape[2]))
        for i in range(batch.shape[1]): # loop over the assets
            for j in range(batch.shape[2]): # loop over the values
                asset = batch[:, i, j]
                transformed_assets = np.zeros((batch_shape[0] - 100, 1, 1))
                for k in range(100, asset.shape[0]): # apply transform on each asset
                    slice = asset[k-100:k].reshape((100))
                    transformed_value = function(slice) #transform
                    transformed_assets[k-100] = transformed_value
                new_assets[:, :, j] = transformed_assets[:, :, 0]
            if i == 0: final_batch = new_assets
            else: final_batch = np.concatenate([final_batch, new_assets], axis=1)
            new_assets = np.zeros((batch_shape[0] - 100, 1, batch_shape[2]))
        return final_batch
        
    @staticmethod
    def keep_every_nth_value(n,arr):
        result = arr[::-n]
        result = result[::-1]
        if len(arr) % n == 0:
            result = np.resize(result, (len(arr) // n,)+result.shape[1:])
        else:
            result = np.resize(result, ((len(arr) + n - 1) // n,)+result.shape[1:])
            result[0] = arr[0]
        return result
    
    @staticmethod
    def fractal_dimension(series, min_window=10):
        def _RS(series):
            pcts = series[1:] / series[:-1] - 1.
            R = max(series) / min(series) - 1.
            S = np.std(pcts, ddof=1)
            return R / S if R and S else 0
        window_sizes = [int(10 ** x) for x in np.arange(np.log10(min_window), np.log10(len(series)), 0.25)] + [len(series)]
        RS = [np.mean(np.apply_along_axis(_RS, 1, np.array([series[start:start+w] for start in range(0, len(series), w) if (start+w) <= len(series)]))) for w in window_sizes]
        H = np.linalg.lstsq(np.column_stack((np.log10(window_sizes), np.ones(len(RS)))), np.log10(RS), rcond=None)[0][0]
        return 2 - H
    
    @staticmethod
    def approximate_entropy(U, lenght_similiar_subsequence=10):
        N = U.shape[0]
        assert N >= 100
        r = int(0.2 * np.std(U))
        def _phi(lenght_similiar_subsequence):
            z = N - lenght_similiar_subsequence + 1.0
            x = np.array([U[i:i+lenght_similiar_subsequence] for i in range(int(z))])
            X = np.repeat(x[:, np.newaxis], 1, axis=2)
            C = np.sum(np.absolute(x - X).max(axis=2) <= r, axis=0) / z
            return np.log(C).sum() / z
        return abs(_phi(lenght_similiar_subsequence + 1) - _phi(lenght_similiar_subsequence))

# #### test code
# env = forecast_crypto(render_mode="human",symb=['ETHUSDT','BTCUSDT','DOGEUSDT'], mode = "live")
# env.reset()
# for i in range(30):
#     action = np.random.choice(list(env._action_to_direction.keys()))
#     env.step(action)
#     env.render()
#     time.sleep(2)
# env.close()



