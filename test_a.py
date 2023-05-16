import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import utils
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from cityscapesscripts.helpers.labels import trainId2label as t2l
import cv2

transform=A.Compose(
[
    A.Resize(256, 512),
    A.HorizontalFlip(),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
]
)
# print(transform)
# print(A.Compose(transform[:-2],ToTensorV2() ))
# exit(0)
train_data = utils.get_cityscapes_data(mode='fine', split='train', num_workers = 4, batch_size = 1, transforms = transform)

img, l, a,b = next(iter(train_data))
print(img.size)
img = torchvision.utils.make_grid(img)
# l = torchvision.utils.make_grid(l)
img = np.transpose(img.numpy(), (1,2,0))



print(l.numpy()[0].shape)
# l.apply_(lambda x: t2l[x].id)

# l = np.transpose(l.numpy(), (1,2,0))
# print(l.numpy()[0].shape)
l = utils.decode_segmap(l.numpy()[0])

# l = l.astype(np.uint8)
# cv2.imshow("R", l)
# cv2.waitKey(0)

plt.figure()
plt.imshow(img)
plt.figure()
plt.imshow(l)
plt.show()