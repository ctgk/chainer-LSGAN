import argparse
import chainer
import chainer.functions as F
import chainer.links as L
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_mldata


class Discriminator(chainer.Chain):

    def __init__(self):
        super(Discriminator, self).__init__(
            l1=L.Linear(784, 400, wscale=0.0013),
            l2=L.Linear(400, 1, wscale=0.0025))

    def __call__(self, x):
        h = F.relu(self.l1(x))
        return self.l2(h)


class Generator(chainer.Chain):

    def __init__(self):
        super(Generator, self).__init__(
            l1=L.Linear(100, 400, wscale=0.01),
            l2=L.Linear(400, 784, wscale=0.0025))

    def __call__(self, x):
        h = F.relu(self.l1(x))
        return F.sigmoid(self.l2(h))


def sample(n):
    return np.float32(np.random.normal(size=(n, 100)))


parser = argparse.ArgumentParser(
    description="Least Square Generative Adversarial Network")
parser.add_argument(
    "--gpu", "-g", type=int, default=-1,
    help="negative value indicates no gpu, default=-1")
args = parser.parse_args()

D = Discriminator()
G = Generator()

if args.gpu >= 0:
    chainer.cuda.get_device(args.gpu).use()
    D.to_gpu()
    G.to_gpu()
    xp = chainer.cuda.cupy
else:
    xp = np

mnist = fetch_mldata("MNIST original")
x = np.float32(mnist.data)
x /= np.max(x, axis=1, keepdims=True)

g_optimizer = chainer.optimizers.Adam()
g_optimizer.use_cleargrads()
g_optimizer.setup(G)
d_optimizer = chainer.optimizers.Adam()
d_optimizer.use_cleargrads()
d_optimizer.setup(D)

for i in range(1, 501):
    for j in range(0, len(x), 100):
        G.cleargrads()
        D.cleargrads()
        g_loss = 0.5 * F.sum(F.square(D(G(xp.asarray(sample(100)))) - 1))
        g_loss.backward()
        g_optimizer.update()
        G.cleargrads()
        D.cleargrads()
        z = np.float32(np.random.normal(size=(100, 100)))
        d_loss = (
            0.5 * F.sum(F.square(D(G(xp.asarray(sample(100))))))
            + 0.5 * F.sum(F.square(D(xp.asarray(x[j: j + 100])) - 1)))
        d_loss.backward()
        d_optimizer.update()
    if i % 10 == 0:
        print("epoch {0:04d}".format(i), end=", ")
        print("g_loss {}".format(g_loss.data), end=", ")
        print("d_loss {}".format(d_loss.data))
        images = G(xp.asarray(sample(25))).data
        if args.gpu >= 0:
            images = chainer.cuda.to_cpu(images)
        for j, img in enumerate(images):
            plt.subplot(5, 5, j + 1)
            plt.imshow(img.reshape(28, 28), cmap="gray")
            plt.axis("off")
        plt.savefig("{0:04d}.png".format(i))
        plt.clf()
