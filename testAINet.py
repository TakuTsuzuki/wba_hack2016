# reference:
# https://github.com/pfnet/chainer/blob/14287ec639405dffb206e9e83f7ace6a7495b9e5/examples/vae/train_vae.py
# http://docs.chainer.org/en/v1.15.0.1/tutorial/recurrentnet.html

import numpy as np
import chainer

import active_inference as ai


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

xp = np
world = np.load('datagen/mnist_matrix/mnist_matrix.npy')

# define network
sensor = 784
hidden = 100
action = 2
encoder = ai.REncoder(sensor, hidden)
decoder = ai.RDecoder(hidden, sensor)
action = ai.Action(sensor, hidden, action)
ainet = ai.ActiveInference(encoder, decoder, action)
fe = ai.FreeEnergy(ainet)
optimizer = chainer.optimizers.Adam()
optimizer.setup(fe)

eye = Eye(world, np.sqrt(sensor))

n_epoch = 10
length_epoch = 100
n_bprop = 10
for epoch in range(1, n_epoch):
    print('epoch', epoch)

    sum_loss = 0
    sum_recon_loss = 0
    dx, dy = 0, 0
    loss = 0
    ainet.reset_state()
    fe.cleargrads()
    for i in range(length_epoch):
        glim = eye.glimpse(dx, dy).reshape(1, sensor).astype(np.float32)
        x = chainer.Variable(xp.asarray(glim))
        loss += fe(x)
        action = ainet.a.data
        dx = int(action[0][0])
        dy = int(action[0][1])
        if (i + 1) % n_bprop == 0:
            fe.cleargrads()
            loss.backward()
            loss.unchain_backward()
            optimizer.update()

        sum_loss += float(fe.loss.data)
        sum_recon_loss += float(fe.recon_loss.data)

    print('mean loss={}, mean reconstruction loss={}'
          .format(sum_loss / length_epoch,
                  sum_recon_loss / length_epoch))
