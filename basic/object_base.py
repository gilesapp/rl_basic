import uuid
import datetime

class bObject():
    def __init__(self):
        self.health = 1
        self.isLive = True
        self.obj_type = 'base'
        
        self.obj_info = {}
        self.obj_id = ''
        self.generate_id()
        
        # display
        self.obj_icon = '\u25CB'
    
    def generate_id(self):
        current_datetime = datetime.datetime.now()
        datetime_string = current_datetime.strftime("%Y%m%d%H%M%S")
        base_uuid = uuid.uuid1()
        self.obj_id = f'{self.obj_type}_{datetime_string}_{base_uuid.fields[0]}'
        
    def live_check(self):
        if self.health <= 0:
            self.destroy()
            return False
        return True
            
    def destroy(self):
        self.isLive = False   
    
    def get_obj_info(self, info_name):
        return ''

class bCharacter(bObject): 
    def __init__(self, tier=1, lv=0, cons=1, dex=1, mind=1, base=1, intel=1):
        super().__init__()
        self.obj_icon = '\U0001F464'
        self.obj_type = 'character'
        
        # main attributes
        self.tier = tier
        self.lv = min(lv, self.tier*10)
        
        self.constitution = min(cons, self.tier*15)
        self.dexterity = min(dex, self.tier*15)
        self.mind = min(mind, self.tier*15)
        self.base = min(base, 9)
        self.intelligence = min(intel, 9)
        
        # general
        self.health = 0
        self.mana = 0
        self.stamina = 0
        
        # physical
        self.physical_damage = 0
        self.physical_defense = 0
        
        # magical
        self.magical_damage = 0
        self.magical_defense = 0
        
        # movement
        self.move_distance = 1
        
        # action
        self.available_action = {}
        
        # init
        self.init_attribute()

    def init_attribute(self):
        # general
        self.health = int(self.constitution * 100)
        self.mana = int(self.base * 10)
        self.stamina = int(self.constitution * 100)
        
        # physical
        self.physical_damage = int(self.constitution * 10)
        self.physical_defense = int(self.constitution * 1)
        
        # magical
        self.magical_damage = int(self.lv * 10)
        self.magical_defense = int(self.constitution * 1)
        
        # movement
        self.move_distance = int(self.dexterity * 1)
        
        # obj info
        self.obj_info['health'] = self.health
        self.obj_info['mana'] = self.mana
        self.obj_info['stamina'] = self.stamina
        self.obj_info['physical_damage'] = self.physical_damage
        self.obj_info['physical_defense'] = self.physical_defense
        self.obj_info['magical_damage'] = self.magical_damage
        self.obj_info['magical_defense'] = self.magical_defense
        
    def get_apply_damage_pack(self, damage_value):
        apply_pack = {
            'source': self,
            'damage_value': damage_value
        }
    
    def can_perform_action(self, action_name):
        return self.available_action.get(action_name, 0) == 1 and self.isLive
        
    def register_action(self, action_name):
        self.available_action[action_name] = 1
    
    def remove_action(self, action_name):
        _ = self.available_action.pop(action_name, None)
        
    def enable_action(self, action_name):
        if self.available_action.get(action_name, None):
            self.available_action[action_name] = 1
    
    def disable_action(self, action_name):
        if self.available_action.get(action_name, None):
            self.available_action[action_name] = 0
    
    def receive_damage(self, damage_pack):
        self.health += damage_pack['damage_value']
        self.live_check()
    
    def get_obj_info(self, info_name):
        return self.obj_info.get(info_name, None)