from dataloaders import get_dataloaders
import matplotlib.pyplot as plt
import numpy as np

train_loader, test_loader = get_dataloaders('mnist_m')

pixels = [0.0, 0.0, 0.0]
count = 0

for batch, _ in train_loader:
    for i in range(batch.shape[0]):
        img = batch[i].numpy().transpose(1, 2, 0)
        curr_sum = np.mean(img, axis=(0, 1))
        for i in range(3):
            pixels[i] += curr_sum[i]
        count += 1

m = (pixels[0] / count, pixels[1] / count, pixels[2] / count)
print('Mean:', m)

pixels = [0.0, 0.0, 0.0]
count = 0
for batch, _ in train_loader:
    for i in range(batch.shape[0]):
        img = batch[i].numpy().transpose(1, 2, 0)
        curr_sum = np.sum((img - m)**2, axis=(0, 1))
        for i in range(3):
            pixels[i] += curr_sum[i]
        count += img.shape[0] * img.shape[1]

s = (np.sqrt(pixels[0] / count), np.sqrt(pixels[1] / count), np.sqrt(pixels[2] / count))
print('std:', s)

# examples = enumerate(test_loader)
# batch_idx, (example_data, example_targets) = next(examples)
#
# fig = plt.figure()
# for i in range(6):
#     plt.subplot(2,3,i+1)
#     plt.tight_layout()
#     # plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
#     plt.imshow(example_data[i].numpy().transpose(1, 2, 0), interpolation='none')
#     plt.title("Ground Truth: {}".format(example_targets[i]))
#     plt.axis('off')
#
# plt.show()