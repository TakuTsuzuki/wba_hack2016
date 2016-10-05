class Eye(object):
    def __init__(self, image, window, init_x=None, init_y=None):
        assert window % 2 == 0, 'window was odd, should be even'
        (h, w) = image.shape
        self.image = image
        self.h = h
        self.w = w
        self.window = window / 2
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
        return self.image[self.y - self.window:self.y + self.window,
                          self.x - self.window:self.x + self.window]

    def check_region(self):
        if self.x - self.window < 0:
            self.x = self.window
        if self.w < self.x + self.window:
            self.x = self.w - self.window
        if self.y - self.window < 0:
            self.y = self.window
        if self.h < self.y + self.window:
            self.y = self.h - self.window


