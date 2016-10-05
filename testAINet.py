# reference:
# https://github.com/pfnet/chainer/blob/14287ec639405dffb206e9e83f7ace6a7495b9e5/examples/vae/train_vae.py
# http://docs.chainer.org/en/v1.15.0.1/tutorial/recurrentnet.html

import numpy as np
import chainer
from chainer import cuda

import active_inference as ai
import body

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

# to gpu
device = 0
xp = cuda.cupy if device >= 0 else np
cuda.get_device(device).use()
fe.to_gpu()

# define body
eye = body.Eye(world, np.sqrt(sensor))

# run simulation
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
