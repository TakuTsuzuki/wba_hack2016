import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L


class Encoder(chainer.Chain):
    def __init__(self, sensor, hidden, top_down=0):
        super(Encoder, self).__init__(
                h_enc=L.LSTM(sensor+top_down, hidden),
                #h_enc=L.Linear(sensor+top_down, hidden),
                z_mu=L.Linear(hidden, hidden),
                z_var=L.Linear(hidden, hidden)
                )

    def __call__(self, x):
        state = x
        r = F.tanh(self.h_enc(state))
        z_mu = self.z_mu(r)
        z_var = self.z_var(r)
        return (z_mu, z_var, state)


class Decoder(chainer.Chain):
    def __init__(self, hidden, sensor, top_down=0):
        super(Decoder, self).__init__(
                h_dec=L.LSTM(hidden, hidden),
                #h_dec=L.Linear(hidden, hidden),
                x_mu=L.Linear(hidden, sensor),
                )

    def __call__(self, z, sigmoid=True):
        x_h = F.tanh(self.h_dec(z))
        x_mu = self.x_mu(x_h)
        if sigmoid:
            return F.sigmoid(x_mu)
        else:
            return x_mu
        return x_mu


class Action(chainer.Chain):
    def __init__(self, state, hidden, action):
        super(Action, self).__init__(
                act_h=L.Linear(state+hidden, state+hidden),
                act_mu=L.Linear(state+hidden, action)
                )
        
    def __call__(self, z, s, sigmoid=True):
        x_h = F.tanh(self.act_h(F.concat((z, s))))
        x_mu = self.act_mu(x_h)
        if sigmoid:
            return F.sigmoid(x_mu)
        else:
            return x_mu
        return x_mu


class ActiveInference(chainer.Chain):
    def __init__(self, encoder, decoder, action):
        super(ActiveInference, self).__init__(
                encoder=encoder,
                decoder=decoder,
                action=action
                )

    def __call__(self, x, top_down=None, sigmoid=True):
        if top_down is not None:
            x = self.xp.concatenate(x, top_down)
        z_mu, z_var, state = self.encoder(x)
        z = F.gaussian(z_mu, z_var)
        x_mu = self.decoder(z, sigmoid)
        a_mu = self.action(z, state, sigmoid)
        self.a_mu = a_mu
        self.a = self.bernoulli(a_mu)
        self.x_mu = x_mu
        return self.a, a_mu, x_mu, z_mu, z_var

    def bernoulli(self, mu):
        noise = self.xp.random.rand(*mu.data.shape)
        return (noise < mu.data).astype('float32')

class FreeEnergy(chainer.Chain):
    def __init__(self, net, base_decay=0.999):
        super(FreeEnergy, self).__init__(net=net)
        self.base_decay = base_decay
        self.base = None
        self.a = None
        self.a_mu = None
        self.x_mu = None
        self.z_mu = None
        self.z_var = None
        self.a_mu = None
        self.recon_loss = None
        self.loss = None

    def __call__(self, x):
        a, a_mu, x_mu, z_mu, z_var = self.net(x, sigmoid=False)

        if self.a is None:
            self.a = a
            self.a_mu = a_mu
            self.x_mu = x_mu
            self.z_mu = z_mu
            self.z_var = z_var
            self.recon_loss = chainer.Variable(self.xp.array(0)
                                               .astype(x[0].data.dtype))
            self.loss = chainer.Variable(self.xp.array(0)
                                         .astype(x[0].data.dtype))
            return self.loss
        recon_loss = F.bernoulli_nll(x, self.x_mu)
        kl_loss = F.gaussian_kl_divergence(self.z_mu, self.z_var)
        expect_loss = recon_loss + kl_loss

        # baseline
        eloss = expect_loss.data
        if self.base is None:
            self.base = eloss
        else:
            self.base = self.base_decay * self.base \
                        + (1-self.base_decay) * eloss
        # action loss
        act_loss = (eloss - self.base) * bernoulli_nll(self.a, self.a_mu)
        
        self.a = a
        self.a_mu = a_mu
        self.x_mu = x_mu
        self.z_mu = z_mu
        self.z_var = z_var
        self.recon_loss = recon_loss
        self.loss = expect_loss
        return expect_loss + act_loss[0]

def bernoulli_nll(x, y):
    return F.sum(F.softplus(y) - x * y, axis=1)

