import numpy as np

def get_tile_locations(map, tile_values):
    tiles = {}
    for t in tile_values:
        tiles[t] = []
    for y in range(len(map)):
        for x in range(len(map[y])):
            tiles[map[y][x]].append((x,y))
    return tiles

def _calc_dist_floor(map, x, y, types):
    for dy in range(len(map)):
        if y+dy >= len(map):
            break
        if map[y+dy][x] in types:
            return dy-1
    return len(map) - 1

def get_floor_dist(map, fromTypes, floorTypes):
    result = 0
    for y in range(len(map)):
        for x in range(len(map[y])):
            if map[y][x] in fromTypes:
                result += _calc_dist_floor(map, x, y, floorTypes)
    return result

def _calc_group_value(map, x, y, types, relLocs):
    result = 0
    for l in relLocs:
        nx, ny = x+l[0], y+l[1]
        if nx < 0 or ny < 0 or nx >= len(map[0]) or ny >= len(map):
            continue
        if map[ny][nx] in types:
            result += 1
    return result

def get_type_grouping(map, types, relLocs, min, max):
    result = 0
    for y in range(len(map)):
        for x in range(len(map[y])):
            if map[y][x] in types:
                value = _calc_group_value(map, x, y, types, relLocs)
                if value >= min and value <= max:
                    result += 1
    return result

def get_changes(map, vertical=False):
    start_y = 0
    start_x = 0
    if vertical:
        start_y = 1
    else:
        start_x = 1
    value = 0
    for y in range(start_y, len(map)):
        for x in range(start_x, len(map[y])):
            same = False
            if vertical:
                same = map[y][x] == map[y-1][x]
            else:
                same = map[y][x] == map[y][x-1]
            if not same:
                value += 1
    return value

def _get_certain_tiles(map_locations, tile_values):
    tiles=[]
    for v in tile_values:
        tiles.extend(map_locations[v])
    return tiles

def _flood_fill(x, y, color_map, map, color_index, passable_values):
    num_tiles = 0
    queue = [(x, y)]
    while len(queue) > 0:
        (cx, cy) = queue.pop(0)
        if color_map[cy][cx] != -1 or map[cy][cx] not in passable_values:
            continue
        num_tiles += 1
        color_map[cy][cx] = color_index
        for (dx,dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx,ny=cx+dx,cy+dy
            if nx < 0 or ny < 0 or nx >= len(map[0]) or ny >= len(map):
                continue
            queue.append((nx, ny))
    return num_tiles

def calc_num_regions(map, map_locations, passable_values):
    empty_tiles = _get_certain_tiles(map_locations, passable_values)
    region_index=0
    color_map = np.full((len(map), len(map[0])), -1)
    for (x,y) in empty_tiles:
        num_tiles = _flood_fill(x, y, color_map, map, region_index + 1, passable_values)
        if num_tiles > 0:
            region_index += 1
        else:
            continue
    return region_index


def run_dikjstra(x, y, map, passable_values):
    dikjstra_map = np.full((len(map), len(map[0])),-1)
    visited_map = np.zeros((len(map), len(map[0])))
    queue = [(x, y, 0)]
    while len(queue) > 0:
        (cx,cy,cd) = queue.pop(0)
        if map[cy][cx] not in passable_values or (dikjstra_map[cy][cx] >= 0 and dikjstra_map[cy][cx] <= cd):
            continue
        visited_map[cy][cx] = 1
        dikjstra_map[cy][cx] = cd
        for (dx,dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx,ny=cx+dx,cy+dy
            if nx < 0 or ny < 0 or nx >= len(map[0]) or ny >= len(map):
                continue
            queue.append((nx, ny, cd + 1))
    return dikjstra_map, visited_map

def calc_longest_path(map, map_locations, passable_values):
    empty_tiles = _get_certain_tiles(map_locations, passable_values)
    final_visited_map = np.zeros((len(map), len(map[0])))
    final_value = 0
    for (x,y) in empty_tiles:
        if final_visited_map[y][x] > 0:
            continue
        dikjstra_map, visited_map = run_dikjstra(x, y, map, passable_values)
        final_visited_map += visited_map
        (my,mx) = np.unravel_index(np.argmax(dikjstra_map, axis=None), dikjstra_map.shape)
        dikjstra_map, _ = run_dikjstra(mx, my, map, passable_values)
        max_value = np.max(dikjstra_map)
        if max_value > final_value:
            final_value = max_value
    return final_value

def calc_certain_tile(map_locations, tile_values):
    return len(_get_certain_tiles(map_locations, tile_values))

def calc_num_reachable_tile(map, map_locations, start_value, passable_values, reachable_values):
    (sx,sy) = _get_certain_tiles(map_locations, [start_value])[0]
    dikjstra_map, _ = run_dikjstra(sx, sy, map, passable_values)
    tiles = _get_certain_tiles(map_locations, reachable_values)
    total = 0
    for (tx,ty) in tiles:
        if dikjstra_map[ty][tx] >= 0:
            total += 1
    return total

def gen_random_map(random, width, height, prob):
    map = random.choice(list(prob.keys()),size=(height,width),p=list(prob.values())).astype(np.uint8)
    return map

def get_string_map(map, tiles):
    int_to_string = dict((i, s) for i, s in enumerate(tiles))
    result = []
    for y in range(map.shape[0]):
        result.append([])
        for x in range(map.shape[1]):
            result[y].append(int_to_string[int(map[y][x])])
    return result

def get_int_prob(prob, tiles):
    string_to_int = dict((s, i) for i, s in enumerate(tiles))
    result = {}
    total = 0.0
    for t in tiles:
        result[string_to_int[t]] = prob[t]
        total += prob[t]
    for i in result:
        result[i] /= total
    return result

def get_range_reward(new_value, old_value, low, high):
    if new_value >= low and new_value <= high and old_value >= low and old_value <= high:
        return 0
    if old_value <= high and new_value <= high:
        return min(new_value,low) - min(old_value,low)
    if old_value >= low and new_value >= low:
        return max(old_value,high) - max(new_value,high)
    if new_value > high and old_value < low:
        return high - new_value + old_value - low
    if new_value < low and old_value > high:
        return high - old_value + new_value - low
