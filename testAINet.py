# reference:
# https://github.com/pfnet/chainer/blob/14287ec639405dffb206e9e83f7ace6a7495b9e5/examples/vae/train_vae.py
# http://docs.chainer.org/en/v1.15.0.1/tutorial/recurrentnet.html

import argparse
import numpy as np
import chainer
from chainer import cuda

import active_inference as ai
import body

parser = argparse.ArgumentParser(description='active inference netwrok')
parser.add_argument('--image', '-i',
                    default='datagen/mnist_matrix/mnist_matrix.npy',
                    help='Path to image file(.npy)')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--sensor', '-x', default=784, type=int,
                    help='Dimension of x(sensor state)')
parser.add_argument('--hidden', '-z', default=100, type=int,
                    help='Dimension of z(hidden state)')
parser.add_argument('--action', '-a', default=2, type=int,
                    help='Dimension of a(action state)')
parser.add_argument('--epoch', default=10, type=int,
                    help='Number of epoch')
parser.add_argument('--scene', default=100, type=int,
                    help='Epoch length (frames)')
parser.add_argument('--bprop', default=10, type=int,
                    help='Back propagation length (frames)')
args = parser.parse_args()

world = np.load(args.image)

# define network
sensor = args.sensor
hidden = args.hidden
action = args.action
encoder = ai.REncoder(sensor, hidden)
decoder = ai.RDecoder(hidden, sensor)
action = ai.Action(sensor, hidden, action)
ainet = ai.ActiveInference(encoder, decoder, action)
fe = ai.FreeEnergy(ainet)
optimizer = chainer.optimizers.Adam()
optimizer.setup(fe)

# to gpu
device = args.gpu
xp = np
if 0 <= device:
    print('Running on GPU')
    xp = cuda.cupy
    cuda.get_device(device).use()
    fe.to_gpu()
else:
    print('Running on CPU')

# define body
eye = body.Eye(world, np.sqrt(sensor))

# run simulation
n_epoch = args.epoch
length_epoch = args.scene
n_bprop = args.bprop
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
