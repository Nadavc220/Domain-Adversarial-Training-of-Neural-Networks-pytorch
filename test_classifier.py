import torch
import torch.nn.functional as F

from dataloaders import get_dataloaders
from classifiers import get_classifier

cuda = True
###################
# get dataloaders #
###################
kwargs = {'num_workers': 8, 'pin_memory': True}
train_loader, test_loader = get_dataloaders('mnist_m', **kwargs)

######################
# Initialize Network #
######################
ckpt_path = '/home/ubuntu/nadav/GradientReversal/weights/mnist_class/99.35/model.pth'
net = get_classifier('mnist')

if cuda:
    net = torch.nn.DataParallel(net, device_ids=[0])
    net = net.cuda()

state = torch.load(ckpt_path)
net.load_state_dict(state)

net.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        if cuda:
            target = target.cuda()
        output = net(data)
        test_loss += F.nll_loss(output, target, size_average=False).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
test_loss /= len(test_loader.dataset)
acc = 100 * correct.cpu().numpy() / len(test_loader.dataset)
print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({}%)\n'.format(
    test_loss, correct, len(test_loader.dataset), acc))
