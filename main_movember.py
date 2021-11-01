from tkinter import *
from PIL import Image
from PIL import ImageTk
import cv2, threading, os, time
from threading import Thread
from os import listdir
from os.path import isfile, join

import random
import dlib
from imutils import face_utils, rotate_bound
import math

def draw_sprite(frame, sprite, x_offset, y_offset):
    (h,w) = (sprite.shape[0], sprite.shape[1])
    (imgH,imgW) = (frame.shape[0], frame.shape[1])

    if y_offset+h >= imgH: #if sprite gets out of image in the bottom
        sprite = sprite[0:imgH-y_offset,:,:]

    if x_offset+w >= imgW: #if sprite gets out of image to the right
        sprite = sprite[:,0:imgW-x_offset,:]

    if x_offset < 0: #if sprite gets out of image to the left
        sprite = sprite[:,abs(x_offset)::,:]
        w = sprite.shape[1]
        x_offset = 0

    for c in range(3):
            frame[y_offset:y_offset+h, x_offset:x_offset+w, c] =  \
            sprite[:,:,c] * (sprite[:,:,3]/255.0) +  frame[y_offset:y_offset+h, x_offset:x_offset+w, c] * (1.0 - sprite[:,:,3]/255.0)
    return frame

def adjust_sprite2head(sprite, head_width, head_ypos, ontop = True):
    (h_sprite,w_sprite) = (sprite.shape[0], sprite.shape[1])
    factor = 1.0*head_width/w_sprite
    sprite = cv2.resize(sprite, (0,0), fx=factor, fy=factor) # adjust to have the same width as head
    (h_sprite,w_sprite) = (sprite.shape[0], sprite.shape[1])

    y_orig =  head_ypos-h_sprite if ontop else head_ypos 
    if (y_orig < 0): #check if the head is not to close to the top of the image and the sprite would not fit in the screen
            sprite = sprite[abs(y_orig)::,:,:] #in that case, we cut the sprite
            y_orig = 0 #the sprite then begins at the top of the image
    return (sprite, y_orig)

def apply_sprite(image, path2sprite,w,x,y, angle, ontop = True):
    sprite = cv2.imread(path2sprite,-1)
    #print sprite.shape
    sprite = rotate_bound(sprite, angle)
    (sprite, y_final) = adjust_sprite2head(sprite, w, y, ontop)
    image = draw_sprite(image,sprite,x, y_final)

def calculate_inclination(point1, point2):
    x1,x2,y1,y2 = point1[0], point2[0], point1[1], point2[1]
    incl = 180/math.pi*math.atan((float(y2-y1))/(x2-x1))
    return incl

def calculate_boundbox(list_coordinates):
    x = min(list_coordinates[:,0])
    y = min(list_coordinates[:,1])
    w = max(list_coordinates[:,0]) - x
    h = max(list_coordinates[:,1]) - y
    return (x,y,w,h)

def get_face_boundbox(points, face_part):
    if face_part == 1:
        (x,y,w,h) = calculate_boundbox(points[17:22]) #left eyebrow
    elif face_part == 2:
        (x,y,w,h) = calculate_boundbox(points[22:27]) #right eyebrow
    elif face_part == 3:
        (x,y,w,h) = calculate_boundbox(points[36:42]) #left eye
    elif face_part == 4:
        (x,y,w,h) = calculate_boundbox(points[42:48]) #right eye
    elif face_part == 5:
        (x,y,w,h) = calculate_boundbox(points[29:36]) #nose
    elif face_part == 6:
        (x,y,w,h) = calculate_boundbox(points[48:68]) #mouth
    return (x,y,w,h)

def cvloop(run_event):
    global panelA
    global SPRITES

    dir_ = "./sprites/flyes/"
    flies = [f for f in listdir(dir_) if isfile(join(dir_, f))] #image of flies to make the "animation"
    i = 0
    video_capture = cv2.VideoCapture(0) #read from webcam
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    print(fps)
    (x,y,w,h) = (0,0,10,10) #whatever initial values

    #Filters path
    detector = dlib.get_frontal_face_detector()

    #Facial landmarks
    print("[INFO] loading facial landmark predictor...")
    model = "filters/shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(model) # link to model: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    set_var = 1
    temp = 0  

    while run_event.is_set(): #while the thread is active we loop
        ret, image = video_capture.read()
        #height, width, layers = image.shape     #EDIT HERE FOR RESIZE
        #print(height,width,layers)
        new_h = 1080
        new_w = 1440
        image = cv2.resize(image, (new_w, new_h))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)
        #print(len(faces))
        if len(faces) != temp: 
            temp = len(faces)
            set_var = 1
        
        apply_sprite(image, "./sprites/topbanner.png",1080,0,200,0)    
        apply_sprite(image, "./sprites/bottomborder1.png",1080,0,800,0)
        apply_sprite(image, "./sprites/text.png",80,600,640,0)

        for face in faces: #if there are faces
            (x,y,w,h) = (face.left(), face.top(), face.width(), face.height())
            # *** Facial Landmarks detection
            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)
            incl = calculate_inclination(shape[17], shape[26]) #inclination based on eyebrows

            # condition to see if mouth is open
            is_mouth_open = (shape[66][1] -shape[62][1]) >= 10 #y coordiantes of landmark points of lips

            if SPRITES[1]:
                (x1,y1,w1,h1) = get_face_boundbox(shape, 6)
                
                if set_var == 1: 
                    mvar = random.choice([1,2,3,4])
                    gvar = random.choice([1,2,3,4])
                    set_var = 0

                if mvar == 1:
                    apply_sprite(image, "./sprites/moustache.png",w1+50,x1-25,y1+10, incl)
                elif mvar == 2: 
                    apply_sprite(image, "./sprites/moustache2.png",w1+65,x1-32,y1+92, incl)
                elif mvar == 3:
                    apply_sprite(image, "./sprites/moustache3.png",w1+60,x1-30,y1+20, incl)
                elif mvar == 4:
                    apply_sprite(image, "./sprites/moustache4.png",w1+110,x1-52,y1+25, incl)

                (x3,y3,_,h3) = get_face_boundbox(shape, 1)
                                
                if gvar == 1:
                    apply_sprite(image, "./sprites/glasses1.png",w,x,y3, incl, ontop = False)
                elif gvar == 2: 
                    apply_sprite(image, "./sprites/glasses2.png",w,x,y3, incl, ontop = False)
                elif gvar == 3:
                    apply_sprite(image, "./sprites/glasses3.png",w,x,y3, incl, ontop = False)
                elif gvar == 4:
                    apply_sprite(image, "./sprites/glasses4.png",w,x,y3, incl, ontop = False)

        # OpenCV represents image as BGR; PIL but RGB, we need to change the chanel order
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # conerts to PIL format
        image = Image.fromarray(image)
        # Converts to a TK format to visualize it in the GUI
        image = ImageTk.PhotoImage(image)

        #cv2.imshow('frame',image)

        # Actualize the image in the panel to show it
        panelA.configure(image=image)
        panelA.image = image

    video_capture.release()

root = Tk()
root.title("Movember Promo")
this_dir = os.path.dirname(os.path.realpath(__file__))
imgicon = PhotoImage(file=os.path.join(this_dir,'imgs/icon.gif'))
root.tk.call('wm', 'iconphoto', root._w, imgicon)

command = lambda: put_sprite(1)

panelA = Label(root)
panelA.pack( padx=10, pady=10)

SPRITES = [0,1,0,0,0]

run_event = threading.Event()
run_event.set()
action = Thread(target=cvloop, args=(run_event,))
action.setDaemon(True)
action.start()

def terminate():
        global root, run_event, action
        print ("Closing thread opencv...")
        run_event.clear()
        time.sleep(1)
        #action.join() #strangely in Linux this thread does not terminate properly, so .join never finishes
        root.destroy()
        print ("All closed! Ciao")

root.protocol("WM_DELETE_WINDOW", terminate)
root.mainloop() #creates loop of GUI