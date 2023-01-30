
import inspect
import os
import shutil
import re

def get_first_class(file_path):
    with open(file_path, 'r') as f:
        file_contents = f.read()
    match = re.search(r'class (\w+)\(', file_contents)
    if match: return match.group(1)
    else: return None
    
def create_env(new_env = "absolute\\path\\of\\new\\env",
               id_name : str = "new_env_sim",
               folder_name : str = "new_env_folder",
               reward_threshold : float = None,
               nondeterministic : bool = False,
               max_episode_steps : int = None,
               order_enforce : bool = True,
               autoreset : bool = False,
               kwargs : dict = {}):

    import gymnasium
    #use new_env path or find the path
    if isinstance( new_env , str):
        if os.path.exists(new_env): src_file = new_env
        else: raise Exception(f"file : {new_env} doesn't exist")
    else: src_file = inspect.getfile(new_env)
    #find path of gymnasium
    folder_path = os.path.dirname(inspect.getfile(gymnasium))
    env_folder = "\\envs\\"
    #create new path to hold new env file
    dst_folder = folder_path + env_folder + folder_name
    #test if folder exist
    if not os.path.exists(dst_folder): os.makedirs(dst_folder)
    else: raise Exception(f"folder : {dst_folder} already exist")
    #paste env file to new directory
    shutil.copy2(src_file, dst_folder)
    ###create entrypoint for register
    folder_holding_new_env = folder_name
    #find file of the new env .py
    name_of_the_new_env = src_file.split("\\")[-1][:-3]
    #find name of the class in the new env .py
    class_name = get_first_class(src_file)
    #complet entrypoint
    entry_point= f"gymnasium.envs.{folder_holding_new_env}.{name_of_the_new_env}:{class_name}"
    #example entrypoint : gymnasium.envs.classic_control.cartpole:CartPoleEnv
    #register the env
    registry_file = folder_path + env_folder + "__init__.py"
    with open(registry_file, "a") as file:
        file.write("\n ")
        file.write(f"\n# New Env : {id_name}")
        file.write("\n# ----------------------------------------")
        file.write("\nregister(")
        file.write(f"\n    id=\"{id_name}\",")
        file.write(f"\n    entry_point=\"{entry_point}\",")
        file.write(f"\n    reward_threshold={reward_threshold },")
        file.write(f"\n    nondeterministic={nondeterministic},")
        file.write(f"\n    max_episode_steps={max_episode_steps},")
        file.write(f"\n    order_enforce={order_enforce},")
        file.write(f"\n    autoreset={autoreset},")
        file.write(f"\n    kwargs={kwargs},")
        file.write(f"\n)")
        file.write("\n ")
        file.close()
    #create init file of the new env 
    init_folder = dst_folder + "\\"  + "__init__.py"
    with open(init_folder, 'w') as file:
        file.write(f"\nfrom gymnasium.envs.{folder_holding_new_env}.{name_of_the_new_env} import {class_name}")
        file.close()
     
# import new_env
# new_env_path = inspect.getfile(new_env) 
# #or  
# new_env_path = "absolute\\path\\of\\new\\env"
# create_env(new_env = new_env_path,
#             id_name = "new_env_sim",
#             folder_name = "new_env_folder",
#             reward_threshold = None,
#             nondeterministic = False,
#             max_episode_steps = None,
#             order_enforce = True,
#             autoreset = False,
#             kwargs = {})
