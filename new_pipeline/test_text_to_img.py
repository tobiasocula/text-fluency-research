text = "qmlsdkjfmlqsdkjfmlqskjdfmliajrpoihazmlfkjmqsldkfjmlsqkdjfmlqskjdfmlqskjfmlkqsjdfmlkqsjdmflkqsjdmflkqjsdmflkjqsdmfkqjsmqmlsdkjfmlqsdkjfmlqskjdfmliajrpoihazmlfkjmqsldkfjmlsqkdjfmlqskjdfmlqskjfmlkqsjdfmlkqsjdmflkqsjdmflkqjsdmflkjqsdmfkqjsm"
text *= 2
print(len(text))
from funcs import text_to_img_new
from pathlib import Path
res = text_to_img_new(text,
                      font_path=Path.cwd() / "arial" / "ARIAL.TTF",
                      font_size=20
                      )
import cv2

cv2.imshow("overlay", res)
cv2.waitKey(0)