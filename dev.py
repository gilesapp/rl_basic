import basic.lib_base as lib
from basic.object_base import *
from basic.world_base import *
import random

if __name__ == "__main__":
    world_arena = bWorld(world_size=(5, 5),  
                         terran_type={'grass': [2, 3], 'wood': [2, 5]}
                        )
    world_arena.set_action_table({'move': ActionMove})
    world_arena.set_terran_table({'grass': TerranStone, 'wood': TerranWood})
    world_arena.reset(35)
    
    char1 = bCharacter()
    char1.register_action('move')
    world_arena.register_obj(char1, co2key([0, 0]))
    world_arena.draw_world()
    
    # world_arena.register_action('move', char1.obj_id, [1, 1])
    # world_arena.step()
    # world_arena.draw_world()
    
    while world_arena.world_time <= 3:
        world_arena.register_action('move', char1.obj_id, [random.randint(0, 1), random.randint(0, 1)])
        world_arena.step()
    world_arena.draw_world()    
    

