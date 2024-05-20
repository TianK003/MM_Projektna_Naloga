debug_level = 0

def log(message):
    if debug_level == 0:
        return
    print(message)

def set_debug_level(level):
    global debug_level
    debug_level = level