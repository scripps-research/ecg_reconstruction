from matplotlib import cm


def get_colors(color_num: int):

    cmap = cm.get_cmap('turbo')
    
    if color_num == 2:
        
        colors = ['#219EBC', '#FFB703']
    
    elif color_num == 3:
        
        colors = ['#219EBC', '#023047', '#FFB703']
        
    elif color_num == 4:
        
        colors = ['#219EBC', '#023047', '#FFB703', '#FB8500']
        
    elif color_num == 5:
        
        colors = ['#8ECAE6', '#219EBC', '#023047', '#FFB703', '#FB8500']
        
    elif color_num == 6:
        
        colors = ['#8ECAE6', '#219EBC', '#023047', '#817425', '#FFB703', '#FB8500']
        
    elif color_num == 7:
        
        colors = ['#8ECAE6', '#219EBC', '#126782', '#023047', '#817425', '#FFB703', '#FB8500']
        
    elif color_num == 8:
        
        colors = ['#8ECAE6', '#219EBC', '#126782', '#023047', '#817425', '#FFB703', '#FD9E02', '#FB8500']
        
    elif color_num == 9:
        
        colors = ['#8ECAE6', '#58B4D1', '#219EBC', '#126782', '#023047', '#817425', '#FFB703', '#FD9E02', '#FB8500']
        
    elif color_num == 10:
        
        colors = ['#8ECAE6', '#58B4D1', '#219EBC', '#126782', '#023047', '#425236', '#817425', '#FFB703', '#FD9E02', '#FB8500']
        
    else:

        colors = [ cmap( (color_idx + 1) / color_num ) for color_idx in range(color_num)]
    
    return colors

def get_hatches(hatch_num: int):
    
    patterns = [ '+++', 'xxx', 'oo', '///' , '**', '\\\\\\' , 'OO', '..', '|||' , '---', '//oo', '\\\\|', '||**', '--\\\\', '++oo', 'xx**', 'oo--', 'OO||', 'OO..', '**--']
    
    return patterns[:hatch_num]

def get_markers(mark_num: int):
    
    markers = ['o', 'v', 'P', 'x', 'd', '<', '>', 'p']
    
    for _ in range(int(mark_num / 8)):
        markers += ['o', 'v', 'P', 'x', 'd', '<', '>', 'p']
    
    return markers[:mark_num]