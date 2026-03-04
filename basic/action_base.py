from .lib_base import *

class bAction():
    def __init__(self):
        self.action_name = 'null'
        self.can_perform = True
        
    def perform(self):
        if self.can_perform:
            print(f"null action")
        
class ActionMove(bAction):
    def __init__(self):
        super().__init__()
        self.action_name = 'move'
        
    def perform(self, env, caster_id, dir):
        if env.objs.get(caster_id, None):  # check if valid obj
            caster_obj = env.objs[caster_id]['instance']
            if not caster_obj.can_perform_action(self.action_name):  # check if instance can perform action
                self.can_perform = False
                print(f"---[debug]--- obj can't perform action {self.action_name}")
            if dir[0] > caster_obj.move_distance or dir[1] > caster_obj.move_distance:  # check if dir is valid
                self.can_perform = False
                print(f"---[debug]--- obj can't move {dir} with ability of {caster_obj.move_distance}")
            
            if self.can_perform:
                # move logic
                obj_key = env.objs[caster_id]['location']
                obj_co = key2co(obj_key)
                
                new_x, new_y = [max(0, min(obj_co[0] + dir[0], env.world_size[0])),
                                max(0, min(obj_co[1] + dir[1], env.world_size[1]))]
                
                # check if traversable
                terran_type = env.grid[new_x][new_y]
                obj_new_key = co2key([new_x, new_y])
                
                if env.terran[terran_type].get_terran_info(obj_new_key, 'traversable') == 1:
                    env.set_obj_info(caster_id, 'location', obj_new_key)
                    
                    env.terran[terran_type].set_terran_info(obj_new_key, 'traversable', 0)
                    env.terran[env.grid[obj_co[0]][obj_co[1]]].set_terran_info(obj_key, 'traversable', 1)