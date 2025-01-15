import cv2 # imports opencv

class ObjectDetection: # A class with methods and attributes to detect objects
    def __init__(self, weights_path = 'dnn_model/yolov4.weights', cfg_path = 'dnn_model/yolov4.cfg',): # Class constructor
        print('Running opencv with YOLOv4')
        # Attributes of the class
        self.nmsThreshold = 0.4 # Set to the recommended value (Non-maximum suppression Threshold)
        self.confThreshold = 0.5 # Set to the recommended value (Confidence Threshold for detections)
        self.image_size = 416 # Set to the recommended value (Size of input images for neural network)
        self.classes = [] # Empty list with classes of objects detected
        self.load_class_names() # Loads the classes names of objects detected
        # Neural network configuration
        net = cv2.dnn.readNet(weights_path, cfg_path) # Loading neural network
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA) # Enables use of NVIDIA GPU with CUDA
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        # Detection model
        self.model = cv2.dnn_DetectionModel(net) # Loads the detection model with the neural network
        self.model.setInputParams(size = (self.image_size, self.image_size), scale = 1/255) # Setting input parameters for model

    def load_class_names(self, classes_path = 'dnn_model/classes.txt'):
        with open (classes_path, 'r') as file_object: # Opens the file with classes as a only reading file
            for class_name in file_object: # Deletes blank spaces and fill the list with the classes
                class_name = class_name.strip('')
                self.classes.append(class_name)
        return self.classes

    def detect(self, frame): # Method of the model which detects objects in a frame
        return self.model.detect(frame, nmsThreshold = self.nmsThreshold, confThreshold = self.confThreshold)
        
            





