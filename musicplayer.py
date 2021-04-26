# ---------------------------------------------- PREDICTION PROGRAM STARTS HERE --------------------------------------------------
from tkinter import *  # tkinter is imported for GUI
from tkinter import filedialog  # for browsing through the files
from tkinter.ttk import Progressbar  # for volume and seek bar
from pygame import mixer  # for different functions like mixer.music.play()
from PIL import Image, ImageTk
import numpy as np
import cv2
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import os
from time import sleep

# from mutagen.mp3 import MP3
# import datetime
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def det_show_emotion(image_path, real_time):
    face_classifier = cv2.CascadeClassifier(
        "/home/artrimiss/Downloads/ml/ML_SA_FINAL/train_model/haarcascade_frontalface_default.xml"
    )
    classifier = load_model(
        "/home/artrimiss/Downloads/ml/ML_SA_FINAL/train_model/Emotion_Detection.h5"
    )

    class_labels = ["Angry", "Happy", "Neutral", "Sad", "surprise"]
    # real_time=0

    if real_time == 0:
        frame = cv2.imread(image_path)
        labels = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, p, q) in faces:
            cv2.rectangle(frame, (x, y), (x + p, y + q), (255, 0, 0), 2)
            roi_gray = gray[y : y + q, x : x + p]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                # make a prediction on the ROI, then lookup the class

                preds = classifier.predict(roi)[0]
                print("prediction = ", preds)
                label = class_labels[preds.argmax()]
                print("prediction max = ", preds.argmax())
                print("label = ", label)
                label_position = (x + 50, y + 30)
                if preds.argmax() == 0 or preds.argmax() == 3:
                    cv2.putText(
                        frame,
                        label,
                        label_position,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (0, 0, 255),
                        3,
                    )
                elif preds.argmax() == 1:
                    cv2.putText(
                        frame,
                        label,
                        label_position,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (0, 255, 0),
                        3,
                    )
                else:
                    cv2.putText(
                        frame,
                        label,
                        label_position,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (255, 0, 0),
                        3,
                    )
                print(os.path.basename(image_path)[:-4] + "detected.jpg")
                os.chdir("/home/artrimiss/Downloads/ml/ML_SA_FINAL/output_images")
                cv2.imwrite(os.path.basename(image_path)[:-4] + "detected.jpg", frame)
            else:
                cv2.putText(
                    frame,
                    "No Face Found",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 255, 0),
                    3,
                )
                print("no face found")
            print("\n")

    else:
        cap = cv2.VideoCapture(0)
        while True:
            # Grab a single frame of video
            ret, frame = cap.read()
            labels = []
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y : y + h, x : x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                if np.sum([roi_gray]) != 0:
                    roi = roi_gray.astype("float") / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)

                    # make a prediction on the ROI, then lookup the class

                    preds = classifier.predict(roi)[0]
                    print("prediction = ", preds)
                    label = class_labels[preds.argmax()]
                    print("prediction max = ", preds.argmax())
                    print("label = ", label)
                    label_position = (x, y)
                    cv2.putText(
                        frame,
                        label,
                        label_position,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (0, 255, 0),
                        3,
                    )
                else:
                    cv2.putText(
                        frame,
                        "No Face Found",
                        (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (0, 255, 0),
                        3,
                    )
                print("\n")
            cv2.imshow("Emotion Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


# ---------------------------------------------- PREDICTION PROGRAM ENDS HERE--------------------------------------------------

# ----------------------------------------------VOLUME FUNCTIONS------------------------------------------
def vup():
    vol = mixer.music.get_volume()
    mixer.music.set_volume(vol + 0.05)
    voltext.configure(text="{}%".format(int(mixer.music.get_volume() * 100)))
    volbar["value"] = mixer.music.get_volume() * 100


def vdown():
    vol = mixer.music.get_volume()
    mixer.music.set_volume(vol - 0.05)
    voltext.configure(text="{}%".format(int(mixer.music.get_volume() * 100)))
    volbar["value"] = mixer.music.get_volume() * 100


def mute():
    global cur_vol
    cur_vol = mixer.music.get_volume()
    mixer.music.set_volume(0)
    root.mute_button.grid_remove()
    root.unmute_button.grid()
    root.status_song.configure(text="MUTED")
    voltext.configure(text="{}%".format(int(mixer.music.get_volume() * 100)))
    volbar["value"] = mixer.music.get_volume() * 100


def unmute():
    root.mute_button.grid()
    root.unmute_button.grid_remove()
    root.status_song.configure(text="Playing..")
    mixer.music.set_volume(cur_vol)
    voltext.configure(text="{}%".format(int(mixer.music.get_volume() * 100)))
    volbar["value"] = mixer.music.get_volume() * 100


# ------------------------------------------------PLAY/PAUSE----------------------------------------
def return_image_path():
    sel_img = filedialog.askopenfilename(title="Select Image file for input")
    image_path.set(sel_img)
    det_show_emotion(sel_img, 0)
    root.input_image = ImageTk.PhotoImage(
        Image.open(sel_img).resize((170, 170), Image.ANTIALIAS)
    )
    root.in_disp_image = Label(root, image=root.input_image)
    root.in_disp_image.grid(row=2, column=2, ipadx=5, ipady=5)
    sleep(0.1)

    root.output_image = ImageTk.PhotoImage(
        Image.open(sel_img).resize((170, 170), Image.ANTIALIAS)
    )
    root.out_disp_image = Label(root, image=root.output_image)
    root.out_disp_image.grid(row=2, column=2, ipadx=5, ipady=5)


def playsong():
    root.status_song.configure(text="PLAYING..")
    seekbody.grid()
    mixer.music.load(songname.get())
    mixer.music.play()
    # song_len=int(MP3(songname.get()).info.length)
    # print(song_len)
    # seekbar['maximum']=song_len
    # endtime.configure(text='{}:{}'.format())


def resumesong():
    mixer.music.unpause()
    root.status_song.configure(text="RESUMED")
    root.resume_button.grid_remove()
    root.pause_button.grid()


def pausesong():
    mixer.music.pause()
    root.status_song.configure(text="PAUSED")
    root.pause_button.grid_remove()
    root.resume_button.grid()


def realtime():
    det_show_emotion("none", 1)


# ---------------------------------------------GUI FUNCTION-------------------------------------------------


def gui():
    global voltext, volbar, status_song, seekbody, song_len, seekbar, endtime
    # -------------------------------------------------LABELS----------------------------------------------
    browse_song = Label(
        root,
        text="Select Image File",
        bg="lawn green",
        font=("Comic Sans MS", 20, "bold"),
    )
    browse_song.grid(row=0, column=0, padx=10, pady=0)

    root.status_song = Label(
        root,
        text="No Track Selected",
        bg="lawn green",
        font=("Comic Sans MS", 20, "bold"),
    )
    root.status_song.grid(row=1, column=2, ipadx=5, ipady=5)

    credit = Label(
        root,
        text="MADE BY SAMAKSH MITTAL(18BEC093)\nAND\nSAMARTH GARG(18BEC094)",
        bg="lawn green",
        font=("arial", 14, "bold"),
    )
    credit.grid(row=3, column=1, padx=0, pady=0, columnspan=3)

    # -------------------------------------------------ENTRIES----------------------------------------------
    song_entry = Entry(
        root, font=("arial", 20, "bold"), width=40, textvariable=image_path
    )
    song_entry.grid(row=0, column=1, columnspan=3, padx=20, pady=0)

    # -------------------------------------------------BUTTONS------------------------------------------
    browse_button = Button(
        root,
        text="DETECT",
        font=("Comic Sans MS", 25, "bold"),
        width=8,
        activebackground="grey30",
        command=return_image_path,
        bd=5,
    )
    browse_button.grid(row=0, column=4, padx=10, pady=20)

    play_button = Button(
        root,
        text="PLAY",
        bg="lawn green",
        font=("Comic Sans MS", 25, "bold"),
        width=8,
        bd=5,
        activebackground="green4",
        command=playsong,
    )
    play_button.grid(row=1, column=0, pady=10)

    root.resume_button = Button(
        root,
        text="RESUME",
        font=("Comic Sans MS", 25, "bold"),
        width=8,
        activebackground="grey30",
        command=resumesong,
        bd=5,
    )
    root.resume_button.grid(row=2, column=0, pady=10)

    root.pause_button = Button(
        root,
        text="PAUSE",
        font=("Comic Sans MS", 25, "bold"),
        width=8,
        bd=5,
        activebackground="grey30",
        command=pausesong,
    )
    root.pause_button.grid(row=2, column=0, pady=10)

    root.real_time_button = Button(
        root,
        text="Real Time",
        font=("Comic Sans MS", 25, "bold"),
        width=8,
        bd=5,
        activebackground="grey30",
        command=realtime,
    )
    root.real_time_button.grid(row=3, column=0)

    vup_button = Button(
        root,
        text="Vol +",
        font=("Comic Sans MS", 25, "bold"),
        width=8,
        bd=5,
        activebackground="grey30",
        command=vup,
    )
    vup_button.grid(row=1, column=4, padx=20, pady=0)

    vdown_button = Button(
        root,
        text="Vol -",
        font=("Comic Sans MS", 25, "bold"),
        width=8,
        bd=5,
        activebackground="grey30",
        command=vdown,
    )
    vdown_button.grid(row=2, column=4, padx=0, pady=0)

    root.unmute_button = Button(
        root,
        text="UNMUTE",
        font=("Comic Sans MS", 25, "bold"),
        width=8,
        bd=5,
        activebackground="grey30",
        command=unmute,
    )
    root.unmute_button.grid(row=3, column=4, padx=0, pady=0)

    root.mute_button = Button(
        root,
        text="MUTE",
        font=("Comic Sans MS", 25, "bold"),
        width=8,
        bd=5,
        activebackground="grey30",
        command=mute,
    )
    root.mute_button.grid(row=3, column=4, padx=0, pady=0)

    # ----------------------------------------------------------VOLUME  BAR-------------------------------------------------
    vollabel = Label(root, text="", bg="red", bd=1)
    vollabel.grid(row=1, column=5, rowspan=3, padx=10, ipadx=0, pady=20)

    volbar = Progressbar(
        vollabel, orient=VERTICAL, mode="determinate", value=100, length=220
    )
    volbar.grid(row=0, column=0, ipadx=8)

    voltext = Label(
        vollabel, text="100%", bg="lightgray", width=4, font=("arial", 10, "bold")
    )
    voltext.grid(row=0, column=0)

    # ----------------------------------------------------------SEEK BAR-------------------------------------------------
    seekbody = Label(root, text="", bg="red", bd=1)
    seekbody.grid(row=2, column=1, columnspan=3, padx=0, pady=0)
    seekbody.grid_remove()

    starttime = Label(seekbody, text="0:00", bg="red", bd=1)
    starttime.grid(row=0, column=0, padx=0, pady=0)

    endtime = Label(seekbody, text="3:00", bg="red", bd=1)
    endtime.grid(row=0, column=3, padx=0, pady=0)

    seekbar = Progressbar(
        seekbody, orient=HORIZONTAL, mode="determinate", value=40, length=530
    )
    seekbar.grid(row=0, column=2, ipady=2)


# -----------------------------------------------------------------MAIN--------------------------------------------------

mixer.init()  # intializing mixer into the program(a function from pygame)
root = Tk()
# 1200X370 is the dimensions of appication dialog box
root.geometry("1150x500+120+150")
# 100 is the margin from left side and 150 is the margin from top
root.title("MUSIC PLAYER")
root.resizable(0, 0)
root.configure(bg="gray25")

image_path = StringVar()
gui()  # calling user defined function gui(all the frontend is in this function)
root.mainloop()  # infinte loop

# ---------------------------------------------- MADE BY SAMAKSH MITTAL ---------------------------------------------------------

# --------------------------------------------------END OF PROGRAM-------------------------------------------------------------------
