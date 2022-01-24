import matplotlib.pyplot as plt
import cv2,sys
from keras.models import load_model

img_original = sys.argv[1].split('=')[1]


def predictMe(img):
    [ans] = model.predict(img)
    arr = ans.copy()
    arr.sort()
    for i in range(len(ans)):
        if arr[-1] == ans[i]:
            print(f"""
[========================]
[   Predicted_Digit={i}    ]
[========================]
		""")



model = load_model('mnist_model.h5')
img_3D = cv2.imread(img_original)

img_2D = cv2.cvtColor(img_3D, cv2.COLOR_BGR2GRAY)
ret,img_2D_bw = cv2.threshold(img_2D,125,255,cv2.THRESH_BINARY)
#plt.imshow(img_2D, cmap='gray')
#plt.show()

img_2D_bw_r = cv2.resize(img_2D_bw,(28,28))
img_i = cv2.bitwise_not(img_2D_bw_r)
img_r = img_i.reshape(1,28*28)
img_r = img_r.astype('float32')
predictMe(img_r)



