import numpy as np
from world import Grass

def padwithtens(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = 0
    vector[-pad_width[1]:] = 0
    return vector

class Body(object):
    def __init__(self, world, window, init_x=None, init_y=None):
        assert window % 2 == 0, 'window was odd, should be even'
        (h, w) = world.shape
        #self.image = image
        self.h = h    # height of pic     
        self.w = w   # wighth of pic
        self.window = window / 2
        self.padwidth = self.window+self.bias
        self.world = np.lib.pad(world, int(self.padwidth), padwithtens)
        if init_x is None:
            self.x = int(h/2) + self.padwidth
        else:
            self.x = init_x
        if init_y is None:
            self.y = int(w/2) + self.padwidth
        else:
            self.y = init_y

    def glimpse(self, dx, dy):
        energy = 0
        if dx == 0:
            dx = -1
            energy = energy + np.sum(self.image[self.y - int(self.window):self.y + int(self.window),
                                                self.x - int(self.window):self.x - int(self.window)+1])
        else:
            energy = energy + np.sum(self.image[self.y - int(self.window):self.y + int(self.window),
                                                self.x + int(self.window) -1:self.x + int(self.window)])
            
        if dy == 0:
            dy = -1
            energy = energy + np.sum(self.image[self.y - int(self.window):self.y - int(self.window)+1,
                                                self.x - int(self.window):self.x + int(self.window)])
        else:
            energy = energy + np.sum(self.image[self.y + int(self.window) -1:self.y + int(self.window),
                                                self.x - int(self.window):self.x + int(self.window)])
            
        
        self.x = self.x + dx
        self.y = self.y + dy
        region = self.image_region()
        
        
        return region, energy

    def image_region(self):
        self.check_region()
        return self.image[self.y - int(self.window):self.y + int(self.window),
                          self.x - int(self.window):self.x + int(self.window)]
    
    def body_region(self):
        return 

    # Make it torus
    def check_region(self):
        self.x=(self.x - self.padwidth) % self.w + self.padwidth
        self.y=(self.y - self.padwidth) % self.h + self.padwidth


