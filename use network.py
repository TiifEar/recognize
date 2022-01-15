import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
from tensorflow import keras
from typing import *
from PIL import ImageTk, ImageDraw, ImageOps, Image
import PIL
from tkinter import *


emnist_labels = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122]

width = 168
height = 168

white = (255, 255, 255)


def emnist_predict_img(model, img):
    img_arr = np.expand_dims(img, axis=0)
    img_arr = 1 - img_arr/255.0
    img_arr[0] = np.rot90(img_arr[0], 3)
    img_arr[0] = np.fliplr(img_arr[0])
    img_arr = img_arr.reshape((1, 28, 28, 1))

    predict_x = model.predict([img_arr])
    classes_x = np.argmax(predict_x, axis=1)
    return chr(emnist_labels[classes_x[0]])

def letters_extract(image_file: str, out_size=28):
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)
    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    output = img.copy()

    letters = []
    for idx, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        print("R", idx, x, y, w, h, cv2.contourArea(contour), hierarchy[0][idx])
        if hierarchy[0][idx][3] == 0:
            cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)
            letter_crop = gray[y:y + h, x:x + w]

            size_max = max(w, h)
            letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
            if w > h:
                y_pos = size_max//2 - h//2
                letter_square[y_pos:y_pos + h, 0:w] = letter_crop
            elif w < h:
                x_pos = size_max//2 - w//2
                letter_square[0:h, x_pos:x_pos + w] = letter_crop
            else:
                letter_square = letter_crop
            letters.append((x, w, cv2.resize(letter_square, (out_size, out_size), interpolation=cv2.INTER_AREA)))
    letters.sort(key=lambda x: x[0], reverse=False)
    return letters

def img_to_str(model: Any, image_file: str):
    letters = letters_extract(image_file)
    if(letters==[]):
        return 0
    s_out = emnist_predict_img(model, letters[0][2])
    return s_out

def clear():
    cv.delete("all")
    lbl1['text'] =''
    draw.rectangle((0, 0, 168, 168), fill=(255, 255, 255, 255))

def paint(event):
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    cv.create_oval(x1, y1, x2, y2, fill="black", width=penSize_slider.get())
    draw.line([x1, y1, x2, y2], fill="black", width=penSize_slider.get())

def recognize():
    #модель нейронної мережі, при бажанні можна замінити
    model = keras.models.load_model('networks/emnist_letters2.h5')
    image.save("converted.png", format="png")
    #якщо потрібно перевірити розпізнавання літери вже записаної в іншому зображенні converted.png замінити на назвузображення і помістити в корінь
    #в тест дата є приклади розпізнавання
    #s_out = img_to_str(model, 'dataset/test_data/h.png')
    s_out = img_to_str(model, 'converted.png')
    lbl1['text'] = s_out

if __name__ == "__main__":
    root = Tk()
    root.title("Розпізнавання літер")

    cv = Canvas(root, width=width, height=height, bg='white')
    cv.pack()

    image = PIL.Image.new("RGB", (width, height), white)
    draw = ImageDraw.Draw(image)

    cv.pack(side=RIGHT)
    cv.bind("<B1-Motion>", paint)

    button = Button(text="розпізнати", command=recognize, width=20)
    button2 = Button(text="очистити", command=clear, width=20)
    lbl0 = Label(text="Розмір", font="Arial 10", width=15)
    lbl1 = Label(text=" ", font="Arial 30", fg="red")
    lbl0.pack()

    penSize_slider = Scale(from_=1, to=10, orient=HORIZONTAL)
    penSize_slider.pack()

    button.pack()
    button2.pack()

    lbl1.pack()
    root.minsize(350, 200)
    root.maxsize(350, 200)

    root.mainloop()