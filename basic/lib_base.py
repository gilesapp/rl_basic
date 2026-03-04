

def cal_apply_damage(damage_type, damage_value, source, target):
    if damage_type == 'physical':
        penetration_factor = source['physical_penetration'] - target['physical_defense']
        if penetration_factor == 0:
            return damage_value * 0.8
        if penetration_factor < 0:
            return damage_value * (1 - min(0.95, penetration_factor * -1 / 10))  # min 5% damage
        if penetration_factor > 0:
            return damage_value

def co2key(xy):
    return f"x{xy[0]}y{xy[1]}"

def key2co(key):
    tmp = key.split("y")
    return (int(tmp[0][1:]), int(tmp[1]))