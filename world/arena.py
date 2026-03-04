from ..basic.object_base import *
from ..basic.world_base import *
    
if __name__ == "__main__":
    world_arena = WorldArena(world_size=(5, 5), 
                             action_table={}, 
                             terran_type={'grass': [2, 3], 'wood': [2, 5]}
                             )
    
    world_arena.draw_world()