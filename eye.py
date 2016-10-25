import numpy as np

def padwithtens(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = 0
    vector[-pad_width[1]:] = 0
    return vector

class Eye(object):
    def __init__(self, image, window, init_x=None, init_y=None):
        assert window % 2 == 0, 'window was odd, should be even'
        (h, w) = image.shape
        #self.image = image
        self.h = h    # height of pic     
        self.w = w   # wighth of pic
        self.window = window / 2
        self.bias =10
        self.padwidth = self.window+self.bias
        self.image = np.lib.pad(image, int(self.padwidth), padwithtens)
        self.time=0.
        if init_x is None:
            self.x = int(h/2) + self.padwidth
        else:
            self.x = init_x
        if init_y is None:
            self.y = int(w/2) + self.padwidth
        else:
            self.y = init_y

    def glimpse(self, dx, dy):
        if dx == 0:
            dx = -1
        if dy == 0:
            dy = -1
        
        self.time = self.time+0.25
        vector=self.bias*np.array((np.cos(self.time/10.),np.cos(self.time/10.)))
        
        self.x = self.x + dx + vector[0]
        self.y = self.y + dy + vector[1]
        return self.region()

    def region(self):
        self.check_region()
        return self.image[self.y - int(self.window):self.y + int(self.window),
                          self.x - int(self.window):self.x + int(self.window)]

    # Make it torus
    def check_region(self):
        self.x=(self.x - self.padwidth) % self.w + self.padwidth
        self.y=(self.y - self.padwidth) % self.h + self.padwidth

"""
class Eye_discrete(object):
    def __init__(self, input_data, init_x=None, init_y=None):
        h = input_data.shape[0]
        w = input_data.shape[1]
        self.image = input_data
        self.h = h    # height of pic     
        self.w = w   # wighth of pic
        #self.window = window / 2
        if init_x is None:
            self.x = int(h/2)
        else:
            self.x = init_x
        if init_y is None:
            self.y = int(w/2)
        else:
            self.y = init_y

    def glimpse(self, dx, dy):
        self.x = self.x + dx
        self.y = self.y + dy
        return self.region()

    def region(self):
        self.check_region()
        return self.image[self.x,self.y,:,:]

    # Make it torus
    def check_region(self):
        self.x=self.x%self.h
        self.y=self.y%self.w 
"""
