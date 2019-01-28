import cv2
import sys
import threading
from PIL import Image
from custom_tf_hub_helper import show_predict_image_objects


facesBox = None
image = None


class MyThread(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)
        self.stopped = False

    def run(self):
        global facesBox

        try:
            while (~(image is None)):
                if self.stopped:
                    return

                facesBox = self.predict(image)
        except Exception as e:
            print(e)
            pass
        

    def predict(self, image):
        if self.stopped:
            return

        if image is not None:
            image_pil = Image.fromarray(image, 'RGB')

            image_with_boxes = show_predict_image_objects(image_pil)

            # print(type(image_with_boxes))
            return image_with_boxes


        return None

    def stop(self):
        self.stopped = True


video_capture = cv2.VideoCapture(0)


this_thread = MyThread()
this_thread.start()


while True:
    ret, image = video_capture.read()

    if facesBox is not None:
        print('facesBox')
        cv2.imshow("Faces found", facesBox)
    else:
        print('normal image')
        cv2.imshow("Faces found", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        this_thread.stop()
        break
