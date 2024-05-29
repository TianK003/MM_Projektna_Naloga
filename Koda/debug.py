import sys

debug_level = 0

def log(message):
    if debug_level == 0:
        return
    print(message)

def set_debug_level(level):
    global debug_level
    debug_level = level

def progress(current, total, without_debug=False):
    if debug_level == 0 and not without_debug:
        return
    
    if current % int(total/100) != 0:
        return
    
    sys.stdout.write('\r')
    percentage = 100 * current / total
    num_symbols = int(percentage/100*20)
    sys.stdout.write("[\033[92m%-20s\033[0m] %d%% " % ('='*(num_symbols), int(percentage)))
    sys.stdout.flush()