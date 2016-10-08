import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L


class REncoder(chainer.Chain):
    def __init__(self, sensor, hidden, top_down=0):
        super(REncoder, self).__init__(
                r_enc=L.LSTM(sensor+top_down, hidden),
                z_mu=L.Linear(hidden, hidden),
                z_var=L.Linear(hidden, hidden)
                )
        self.reset_state()

    def reset_state(self):
        self.r_enc.reset_state()

    def __call__(self, x):
        state = x
        r = self.r_enc(state)
        z_mu = self.z_mu(r)
        z_var = self.z_var(r)
        return (z_mu, z_var, state)


class RDecoder(chainer.Chain):
    def __init__(self, hidden, sensor, top_down=0):
        super(RDecoder, self).__init__(
                r_dec=L.LSTM(hidden, hidden),
                x_mu=L.Linear(hidden, sensor),
                x_var=L.Linear(hidden, sensor),
                )
        self.reset_state()

    def reset_state(self):
        self.r_dec.reset_state()

    def __call__(self, z):
        x_h = self.r_dec(z)
        x_mu = self.x_mu(x_h)
        x_var = self.x_var(x_h)
        return (x_mu, x_var)


class Action(chainer.Chain):
    def __init__(self, state, hidden, action):
        super(Action, self).__init__(
                act_h=L.Linear(state+hidden, state+hidden),
                act_mu=L.Linear(state+hidden, action),
                act_var=L.Linear(state+hidden, action)
                )

    def __call__(self, z, s):
        a_h = self.act_h(F.concat((z, s)))
        a_mu = self.act_mu(a_h)
        a_var = 2 * F.sigmoid(self.act_var(a_h)) - 1
        return (a_mu, a_var)


class ActiveInference(chainer.Chain):
    def __init__(self, encoder, decoder, action):
        super(ActiveInference, self).__init__(
                encoder=encoder,
                decoder=decoder,
                action=action
                )
        self.a = None
        self.x_hat = None
        self.reset_state()

    def reset_state(self):
        self.encoder.reset_state()
        self.decoder.reset_state()

    def __call__(self, x, top_down=None):
        if top_down is not None:
            x = self.xp.concatenate(x, top_down)
        (z_mu, z_var, state) = self.encoder(x)
        z = F.gaussian(z_mu, z_var)
        (x_mu, x_var) = self.decoder(z)

        (a_mu, a_var) = self.action(z, state)
        a = F.gaussian(a_mu, a_var)
        self.a = a
        self.x_hat = x_mu

        return a, (x_mu, x_var), (z_mu, z_var), (a_mu, a_var)


class FreeEnergy(chainer.Chain):
    def __init__(self, net, base_decay=0.999):
        super(FreeEnergy, self).__init__(net=net)
        self.base_decay = base_decay
        self.base = None
        self.a = None
        self.theta_x = None
        self.theta_z = None
        self.theta_a = None
        self.recon_loss = None
        self.loss = None

    def __call__(self, x):
        a, (x_mu, x_var), (z_mu, z_var), (a_mu, a_var) = self.net(x)

        if self.a is None:
            self.a = a
            self.theta_x = (x_mu, x_var)
            self.theta_z = (z_mu, z_var)
            self.theta_a = (a_mu, a_var)
            self.recon_loss = chainer.Variable(self.xp.array(0)
                                               .astype(x[0].data.dtype))
            self.loss = chainer.Variable(self.xp.array(0)
                                         .astype(x[0].data.dtype))
            return self.loss

        recon_loss = F.gaussian_nll(x, self.theta_x[0], self.theta_x[1])
        kl_loss = F.gaussian_kl_divergence(self.theta_z[0], self.theta_z[1])
        expect_loss = recon_loss + kl_loss

        # baseline
        eloss = expect_loss.data
        if self.base is None:
            self.base = eloss
        else:
            self.base = self.base_decay * self.base \
                        + (1-self.base_decay) * eloss
        # action loss
        act_loss = (eloss - self.base) \
            * F.gaussian_nll(chainer.Variable(self.a.data),
                             self.theta_a[0], self.theta_a[1])

        self.theta_x = (x_mu, x_var)
        self.theta_z = (z_mu, z_var)
        self.theta_a = (a_mu, a_var)
        self.recon_loss = recon_loss
        self.loss = expect_loss
        return expect_loss + act_loss
