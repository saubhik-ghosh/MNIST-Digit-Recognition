import pygame, sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2

Windowsizex = 640
Windowsizey = 480

boundry = 5
white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)

imagesave = False

model = load_model("F:/CMREC/Project/3rd Year/Semester 2/Mini Project/MNIST Cust/mnist.h5")

labels = {0: "Zero", 1: "One",
        2: "Two", 3: "Three",
        4: "Four", 5: "Five",
        6: "Six", 7: "Seven",
        8: "Eight", 9: "Nine"}

pygame.init()

FONT = pygame.font.Font("freesansbold.ttf", 18)
Displaysurf = pygame.display.set_mode((Windowsizex, Windowsizey))

pygame.display.set_caption("Digit Board")

iswriting = False

num_xcord = []
num_ycord = []

img_cnt = 1
predict = True

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(Displaysurf, white, (xcord, ycord), 4, 0)

            num_xcord.append(xcord)
            num_ycord.append(ycord)

        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == MOUSEBUTTONUP:
            iswriting = False
            num_xcord = sorted(num_xcord)
            num_ycord = sorted(num_ycord)
            
            rect_min_x, rect_max_x = max(num_xcord[0]-boundry, 0), min(Windowsizex, num_xcord[-1]+boundry)
            rect_min_y, rect_max_y = max(num_ycord[0]-boundry, 0), min(num_ycord[-1]+boundry, Windowsizey)

            pygame.draw.rect(Displaysurf, red, (rect_min_x, rect_min_y, rect_max_x - rect_min_x, rect_max_y - rect_min_y), 2)
            
            num_xcord = []
            num_ycord = []

            img_arr = np.array(pygame.PixelArray(Displaysurf))[rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype(np.float32)

            if imagesave:
                cv2.imwrite("image.png")
                img_cnt += 1 
            
            if predict:
                image = cv2.resize(img_arr, (28, 28))
                image = np.pad(image, (10, 10), 'constant', constant_values=0)
                image = cv2.resize(image, (28, 28))/255

                label = str(labels[np.argmax(model.predict(image.reshape(1, 28, 28, 1)))])

                textsurf = FONT.render(label, True, red, white)
                textrect = textsurf.get_rect()
                textrect.left, textrect.bottom = rect_min_x, rect_max_y

                Displaysurf.blit(textsurf, textrect)
            
            if event.type == KEYDOWN:
                if event.unicode == "n":
                    Displaysurf.fill(black)
        pygame.display.update()