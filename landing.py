import cv2
import numpy as np
img = cv2.imread(r"C:\Temp\landingAI\data\blister_defects_split\ASLK-YB1-0121\pascal_voc_test\Segmentations\2.png")

img.shape

# from matplotlib import pyplot as plt
# plt.imshow(img)
# plt.show()

import numpy as np
np.unique(img[:,:,0], return_counts=True)
np.unique(img[:,:,1], return_counts=True)
np.unique(img[:,:,2], return_counts=True)

np.alltrue(img[:,:,0] == img[:,:,1])

np.alltrue(img[:,:,1] == img[:,:,2])

cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.resizeWindow('img', 800, 600)
cv2.imshow("img",img*255)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.resizeWindow('img', 800, 600)
cv2.imshow("img",img[:,:,0]*255)
cv2.waitKey(0)
cv2.destroyAllWindows()

def plot(x):
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img', 800, 600)
    cv2.imshow("img", x[:, :, 0] * 255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


mask = np.load(r"C:\Temp\landingAI\data\blister_defects_split\ASLK-YB1-0121\pascal_voc_test\id_labels\2.npz")['arr_0']

mask.shape

np.unique(mask)


img = np.zeros((512, 512, 3), dtype=np.uint8)
def r():
    return int(300*np.random.random(1))
img[img==0] = 255
for i in range(1,11):
    print(i)
    img[0, 0, 0] = i

    cv2.imwrite(f"C:/projects/landing/img/{i}.png", img)

def count(x):
    return np.unique(x, return_counts=True)

i = 4
mask = np.zeros((512, 512), dtype=np.uint8)
x = r()
y = r()
print(x, y)
mask[x:(x+50),y:(y+50)] = 1
x = r()
y = r()
mask[x:(x+50),y:(y+50)] = 2
x = r()
y = r()
mask[x:(x+50),y:(y+50)] = 3
cv2.imwrite(f"C:/projects/landing/mask/{i}.png", mask)






np.unique(mask, return_counts=True)

np.unique(cv2.imread(f"C:/projects/landing/1_m.png"), return_counts=True)


x = np.array([
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
], np.uint8)

x[0:1, 0:1] = 100

x





img = cv2.imread(r"C:\Temp\tar\a_first_data_set_2022_10_14-166573883916\test\Segmentations\2022-10-14T08-49-04-371Z-0f1b1255-a9d7-43da-8c2b-39a517283f8f.png")
img.shape
count(img)





import numpy as np
img = cv2.imread(r"C:\Temp\landingAI\data\blister_defects_split\ASLK-YB1-0121\pascal_voc_test\Segmentations\2.png")
img.shape
count(img)













####################################

from landinglens import LandingLens


llens = LandingLens()

media_list = llens.media.ls()

media_list.keys()

media_list['medias']

len(media_list['medias'])

media_list['medias'][0]

# {'id': 7669399,
#  'mediaType': 'image',
#  'srcType': 'user',
#  'srcName': 'Jiayi',
#  'properties': {'width': 2413, 'height': 1087},
#  'name': '02343e03-8cc0-4976-8136-607e264c096b.bmp',
#  'uploadTime': '2022-10-13T07:18:24.440Z',
#  'mediaStatus': 'approved',
#  'metadata': {}}




