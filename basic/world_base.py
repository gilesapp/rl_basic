import random
from .lib_base import *
from .terran_base import *
from .action_base import *

class bWorld:
    def __init__(self, world_size=(1, 1), terran_type={'grass': [2, 5]}):
        self.world_size = world_size  # (1, 1)
        self.terran_type = terran_type  # {terran name: [min size, max size]}
        
        # grid (run time)
        self.grid = []  # [['grass', 'wood'], ['wood', 'grass']]
        self.grid_occupied = set()  # ((0, 0), (0, 1))
        
        # table (static)
        self.action_table = {}  # {action name: action instance}
        self.terran_table = {}  # {terran name: terran class}
        
        # run time
        self.objs = {}  # {id: {*kw}}}
        self.action = {}  # {action name: action instance}
        self.terran = {}  # {terran name: terran instance}
        self.action_buffer = []  # [[action name, *args]]
        self.world_time = 0
    
    def register_action(self, action_name, *args):
        self.action_buffer.append([action_name, *args])
        
    def register_obj(self, obj, key):
        self.objs[obj.obj_id] = {
            'instance': obj,
            'location': key
            }
        
    def set_action_table(self, action_table):
        self.action_table = action_table
        
    def set_terran_table(self, terran_table):
        self.terran_table = terran_table
    
    def create_action(self):
        self.action = {}
        assert len(list(self.action_table.keys())) > 0, "empty action table. e.g. {'attack': ActionAttack}"
        for action_n in self.action_table.keys():
            self.action[action_n] = self.action_table[action_n]()  # create action instance
            
    def create_terran(self):
        self.terran = {}
        assert len(list(self.terran_type.keys())) > 0, "terran type invalid. e.g. {'grass': [2, 5]}"
        assert len(list(self.terran_table.keys())) > 0, "empty terran table. e.g. {'grass': TerranGrass}"
        
        for terran_t in self.terran_type.keys():
            terran_coordinate = []
            for i in range(self.world_size[0]):
                for j in range(self.world_size[1]):
                    if self.grid[i][j] == terran_t:
                        terran_coordinate.append(co2key([i, j]))
            self.terran[terran_t] = self.terran_table[terran_t](terran_coordinate)  # create terran instance 
        
    def can_place(self, x, y, size):
        if x + size > self.world_size[0] or y + size > self.world_size[1]:
            return False
        for i in range(x, x + size):
            for j in range(y, y + size):
                if (i, j) in self.grid_occupied:
                    return False
        return True
    
    def place_terrain(self, x, y, size, terran_type):
        for i in range(x, x + size):
            for j in range(y, y + size):
                self.grid[i][j] = terran_type
                self.grid_occupied.add((i, j))
    
    def create_grid(self, seed=42, fill_up=True):
        assert self.world_size[0] > 1 and self.world_size[1] > 1, "world size invalid. e.g. (1, 1)"
        assert len(list(self.terran_type.keys())) > 0, "terran type invalid. e.g. {'grass': [2, 5]}"
        
        self.grid = [[None for _ in range(self.world_size[0])] for _ in range(self.world_size[1])]
        self.grid_occupied = set()
        
        local_random = random.Random(seed)
        
        attempts = 0
        max_attempts = self.world_size[0] * self.world_size[1] * 3
        
        while len(self.grid_occupied) < self.world_size[0] * self.world_size[1] and attempts < max_attempts:
            attempts += 1
            
            # random terran type and size
            terrain = local_random.choice(list(self.terran_type.keys()))
            size = local_random.randint(self.terran_type[terrain][0], self.terran_type[terrain][1])
            
            # random x, y (start loc: (0, 0))
            x = local_random.randint(0, self.world_size[0] - 1)
            y = local_random.randint(0, self.world_size[1] - 1)
            
            # place terran
            if self.can_place(x, y, size):
                self.place_terrain(x, y, size, terrain)
        
        # fill up gaps (can disable)
        if fill_up:
            for i in range(self.world_size[0]):
                for j in range(self.world_size[1]):
                    if (i, j) not in self.grid_occupied:
                        terrain = local_random.choice(list(self.terran_type.keys()))
                        self.grid[i][j] = terrain
                        self.grid_occupied.add((i, j))
    
    def get_obj_info(self, obj_id, info_name):
        return self.objs[obj_id].get(info_name, None)
    
    def set_obj_info(self, obj_id, info_name, info_value):
        if self.objs[obj_id].get(info_name, None):
            self.objs[obj_id][info_name] = info_value
            
    def reset(self, seed=42, fill_up=True):
        self.objs = {}
        self.create_grid(seed, fill_up)  # init world
        self.create_action()
        self.create_terran()
        
    def step(self):
        # update terran
        for terran_ins in self.terran.values():
            terran_ins.update_terran(self.world_time)
            
        # update objs & actions # [[action name, *args]]
        for action_pack in self.action_buffer:
            self.action[action_pack[0]].perform(self, *action_pack[1:])
        
        self.world_time += 1
        self.action_buffer.clear()
            
    def simulate_step(self):
        pass
    
    def get_obs(self, obj_id, info_list):
        obs = {}
        if self.objs.get(obj_id, None):
            for info_name in info_list:
                obs[info_name] = self.objs[obj_id]['obj'].get_obj_info(info_name)
        
        return obs
    
    # display
    def draw_world(self):
        symbols = {
            'null': '\u2B1C',
            'grass': '\U0001F33F',
            'water': '\U0001F4A7', 
            'wood': '\U0001FAB5'
        }

        draw_obj = {}
        draw_obj_key = []
        for obj_info in self.objs.values():
            draw_obj_key.append(obj_info['location'])
            draw_obj[obj_info['location']] = obj_info['instance'].obj_icon
        
        print("=" * (self.world_size[1] * 2 + 1))
        for i in range(self.world_size[0]):
            line = "|"
            for j in range(self.world_size[1]):
                cur_key = co2key([i, j])
                if cur_key in draw_obj_key:
                    line += draw_obj[cur_key]
                else:
                    line += symbols.get(self.grid[i][j], '\u2B1C')
            print(line + "|")
        print("=" * (self.world_size[1] * 2 + 1))
        
        # print("\ntext：")
        # for row in self.grid:
        #     print([cell if cell else 'None' for cell in row])
        
        
class WorldArena(bWorld):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        
    def step(self):
        # obs, reward, terminated, truncated, info = step()
        return None
    
    def simulate_step(self):
        return None