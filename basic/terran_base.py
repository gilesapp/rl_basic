import random

class bTerran:
    def __init__(self, terran_coordinate):
        self.terran_type = 'null'
        self.terran_coordinate = terran_coordinate
        
        self.terran_info = {}
        self.update_period = 0
        
        self.reset_terran()
    
    def reset_terran(self):
        for terran_co in self.terran_coordinate:
            self.terran_info[terran_co] = {'traversable': 1}
    
    def get_terran_info(self, key_co, info_name):
        return self.terran_info[key_co].get(info_name, None)
    
    def set_terran_info(self, key_co, info_name, info_value):
        if self.terran_info[key_co].get(info_name, None):
            self.terran_info[key_co][info_name] = info_value
    
    def update_terran(self, world_time):
        pass
        
class TerranWood(bTerran):
    def __init__(self, *args, wealthy=2):
        self.wealthy = wealthy
        
        super().__init__(*args)
        self.terran_type = 'wood'
        self.update_period = 60

    def reset_terran(self):
        for terran_co in self.terran_coordinate:
            self.terran_info[terran_co] = {
                'traversable': 1,
                'resource': random.randint(self.wealthy * 10 * 0.8, self.wealthy * 10 * 1.2)
                }    

    def update_terran(self, world_time):
        if world_time % self.update_period == 0:
            for terran_co in self.terran_coordinate:
                self.terran_info[terran_co]['resource'] += 1
            
class TerranStone(bTerran):
    def __init__(self, *args, wealthy=1):
        self.wealthy = wealthy
        
        super().__init__(*args)
        self.terran_type = 'stone'

    def reset_terran(self):
        for terran_co in self.terran_coordinate:
            self.terran_info[terran_co] = {
                'traversable': 1,
                'resource': random.randint(self.wealthy * 10 * 0.7, self.wealthy * 10 * 1.1)
                }  