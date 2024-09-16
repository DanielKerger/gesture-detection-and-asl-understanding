import cv2
import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

IMAGE_SIZE = (200, 200)
MODEL_PATH = "mobilenetv3large-finetune-aug.h5"

class App:
    def __init__(self, window, window_title, video_source=0):
        """Initializes the app with the window, window title, and video source. Also initializes the class labels, model, frame, detected word, and window size.

        Args:
            window: The window to display the app.
            window_title: The title of the window.
            video_source: The video source to capture frames from. Default is 0.

        Returns:
            None    
        """
        self.class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']
        self.model = tf.keras.models.load_model(MODEL_PATH)
        self.frame = 0
        self.detected_word = ""
        self.window = window
        self.window.title(window_title)
        self.window.geometry("1000x800")
        
        self.vid = cv2.VideoCapture(video_source)
        
        self.canvas = ctk.CTkCanvas(window, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.grid(row=0, column=0)
        self.image_container = ctk.CTkImage(light_image=Image.open('hand_bb.jpg'),
            dark_image=Image.open('hand_bb.jpg'),
            size=(640, 480))
        self.image_frame = ctk.CTkLabel(window, image=self.image_container, text="")
        self.image_frame.grid(row=0, column=1)

        self.frame_interval = 72
        self.text_box = ctk.CTkTextbox(window, width=500, height=300)
        self.text_box.grid(row=1, column=0)
        
        self.update()
        
        self.window.mainloop()
        
    def update(self):
        """
        Updates the frame, image, and prediction. Also updates the top 5 predictions and the text box.

        Args:
            None

        Returns:
            None
        """
        ret, image_frame = self.vid.read()
        if ret:
            self.photo = ImageTk.PhotoImage(image = Image.fromarray(cv2.cvtColor(image_frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=ctk.NW)

        if self.frame % self.frame_interval == 0:  
            image_frame = self.detect_hand(image_frame)
            if image_frame is not None:
                image_preprocessed = self.preprocess(image_frame)
                self.image_frame.configure(image=ctk.CTkImage(light_image=Image.open('hand_bb.jpg'), dark_image=Image.open('hand_bb.jpg'), size=(500, 390)))
                self.predict(image_preprocessed)
        self.frame += 1

        self.window.after(10, self.update)

    def preprocess(self, image):
        """
        Preprocesses the image by resizing and expanding the dimensions.
        
        Args:
            image: The image to preprocess.

        Returns:
            The preprocessed image.
        """
        image = cv2.resize(image, IMAGE_SIZE)
        image = np.expand_dims(image, axis=0)
        return image

    def predict(self, image):
        """
        Predicts the image using the model and updates the top 5 predictions and text box.
        
        Args:
            image: The image to predict.
            
        Returns:
            None
        """
        prediction = self.model.predict(image)
        self.update_top5(prediction)
        self.update_text(np.argmax(prediction))

    def update_top5(self, prediction):
        """
        Updates the top 5 predictions using matplotlib and tkinter canvas.

        Args:
            prediction: The prediction to update.

        Returns:
            None
        """
        top5 = np.argsort(prediction[0])[-5:]
        top5_percent = [round(prediction[0][i]*100, 2) for i in top5]
        plt.style.use('dark_background')
        # create matplotlib figure
        fig, ax = plt.subplots(figsize=(6, 3.7))
        ax.barh([self.class_labels[i] for i in top5], top5_percent)
        ax.set_xlabel('Probability (%)')
        ax.set_title('Top 5 Predictions')
        # add lables to bars
        for i, v in enumerate(top5_percent):
            ax.text(v + 1, i, str(v), color='white', va='center')

        # convert figure to tkinter canvas
        canvas = FigureCanvasTkAgg(fig, master=self.window)
        canvas.draw()
        canvas.get_tk_widget().grid(row=1, column=1)
        self.window.update()

    def update_text(self, prediction):
        """
        Updates the text box with the predicted letter.

        Args:
            prediction: The prediction to update.

        Returns:
            None
        """
        if self.class_labels[prediction] == self.class_labels[-1]:
            self.text_box.insert("end", " ")
        elif self.class_labels[prediction] == self.class_labels[-2]:
            pass
        elif self.class_labels[prediction] == self.class_labels[-3]:
            self.text_box.delete("1.0", "end")
        else:
            self.text_box.insert("end", self.class_labels[prediction])

    def detect_hand(self, image):
        """
        Detects the hand in the image and crops the image to the hand bounding box.

        Args:
            image: The image to detect the hand.

        Returns:
            The cropped image.
        """
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(base_options=base_options,
                                            num_hands=1)

        detector = vision.HandLandmarker.create_from_options(options)

        image_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        detection_result = detector.detect(image_mp)

        min_x = 1
        min_y = 1
        max_x = 0
        max_y = 0

        for hand_landmarks in detection_result.hand_landmarks:
            for landmark in hand_landmarks:
                if landmark.x < min_x:
                    min_x = landmark.x
                if landmark.y < min_y:
                    min_y = landmark.y
                if landmark.x > max_x:
                    max_x = landmark.x
                if landmark.y > max_y:
                    max_y = landmark.y

            print(f"min_x: {min_x}, min_y: {min_y}, max_x: {max_x}, max_y: {max_y}")

            # crop image with 10% padding
            padding = 0.1
            min_x = int(min_x*image.shape[1] - padding*image.shape[1])
            min_y = int(min_y*image.shape[0] - padding*image.shape[0])
            max_x = int(max_x*image.shape[1] + padding*image.shape[1])
            max_y = int(max_y*image.shape[0] + padding*image.shape[0])

            # check if min_x and min_x are negative. check if max_x and max_y are greater than image shape. then set them to image border
            if min_x < 0:
                min_x = 0
            if min_y < 0:
                min_y = 0
            if max_x > image.shape[1]:
                max_x = image.shape[1]
            if max_y > image.shape[0]:
                max_y = image.shape[0]

            # create copy image with bounding box
            image_bb = image.copy()
            cv2.rectangle(image_bb, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 255, 0), 2)

            # crop image
            image = image[min_y:max_y, min_x:max_x]
            # export cropped image
            cv2.imwrite("hand_bb.jpg", image_bb)
            return image
        
        return None

App(ctk.CTk(), "ASL Translator")
