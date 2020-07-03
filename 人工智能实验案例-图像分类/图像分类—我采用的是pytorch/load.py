import torchvision
import torchvision.transforms as transforms
import fz_pytorch as fz
mnist_train = torchvision.datasets.FashionMNIST(root='/softmax', train=True, download=True, transform=transforms.ToTensor())
mnist_test =torchvision.datasets.FashionMNIST(root='/softmax', train=False, download=True, transform=transforms.ToTensor())
print(type(mnist_train))
print(len(mnist_train), len(mnist_test))
feature, label = mnist_train[0]
print(feature.shape, label)
X, y = [], []
for i in range(10):
    X.append(mnist_train[i][0])
    y.append(mnist_train[i][1])
fz.show_fashion_mnist(X, fz.get_fashion_mnist_labels(y))