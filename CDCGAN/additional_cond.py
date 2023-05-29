#additional_cond
import numpy as np

def count_sky_tiles(joined_frame):
    level = np.argmax(joined_frame[:, :, 9:-9, 2:-2].data.cpu().numpy(), axis=1)
    sky_region, upper_half = level[:,:10,:], level[:,:7,:]
    ground_tiles = np.logical_or(level == 0, level == 1)

    #num_enemies = np.sum(np.sum(np.logical_and(level[:-1] == 5, ground_tiles[1:]),axis = 1),axis=1)
    num_sky_tiles = np.sum(np.sum(np.logical_and(sky_region != 2, sky_region != 5),axis=1),axis=1)
    num_sky_tiles_at_half = np.sum(np.sum(np.logical_and(upper_half != 2, upper_half != 5),axis=1),axis=1)
    num_ground_tiles = np.sum(np.sum(level == 0,axis=1),axis=1)
    num_total_enemies = np.sum(np.sum(np.logical_or(level == 5, level == 11, level == 12),axis=1),axis=1)
    
    fitness = (num_sky_tiles_at_half + num_sky_tiles)- num_ground_tiles + 10*num_total_enemies
    true = np.where(fitness>=0)
    false = np.where(fitness < 0)

    return true, false


def count_ground_tiles(joined_frame):
    level = np.argmax(joined_frame[:, :, 9:-9, 2:-2].data.cpu().numpy(), axis=1)
    sky_region, upper_half = level[:,:10,:], level[:,:7,:]
    ground_tiles = np.logical_or(level == 0, level == 1)

    #num_enemies = np.sum(np.sum(np.logical_and(level[:-1] == 5, ground_tiles[1:]),axis = 1),axis=1)
    num_sky_tiles = np.sum(np.sum(np.logical_and(sky_region != 2, sky_region != 5),axis=1),axis=1)
    num_sky_tiles_at_half = np.sum(np.sum(np.logical_and(upper_half != 2, upper_half != 5),axis=1),axis=1)
    num_ground_tiles = np.sum(np.sum(level == 0,axis=1),axis=1)
    num_total_enemies = np.sum(np.sum(np.logical_or(level == 5, level == 11, level == 12),axis=1),axis=1)
    
    fitness = -(num_sky_tiles) + num_ground_tiles - 10*num_total_enemies
    true = np.where(np.logical_and(fitness>=5, num_ground_tiles <= 26))
    false = np.where(np.logical_or(fitness < 5, num_ground_tiles > 26))

    return true, false