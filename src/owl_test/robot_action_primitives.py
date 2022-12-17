import torch
import clip
from PIL import Image
import numpy as np


class ObjectFinder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # load model and image preprocessing
        self.model, self.preprocess = clip.load("ViT-B/32", device=device, jit=False)
    def segment_on_table_objects(self, image):
        objects_on_table_roi = []
        center_locations = []
        return objects_on_table_roi, center_locations

    def find_object(self, object_names, image, depth_image):
        objects_on_table_roi, center_loc = self.segment_on_table_objects(depth_image)
        objects_images = []
        for object_roi in objects_on_table_roi:
            objects_images.append(image[object_roi[0]:object_roi[1], object_roi[2]:object_roi[3]])

        text_snippets = ["a photo of a {}".format(name) for name in object_names]
        text_snippets.append("a photo of something else")
        # pre-process text
        text = clip.tokenize(text_snippets).to(device)
        
        # with torch.no_grad():
        #     text_features = model.encode_text(text)
        detected_objects = [None] * len(object_names)
        for i, object_image in enumerate(objects_images):
            # pre-process image
            prepro_image = self.preprocess(object_image).unsqueeze(0).to(self.device)
            
            # with torch.no_grad():
            #     image_features = model.encode_image(prepro_image)
            
            with torch.no_grad():
                logits_per_image, logits_per_text = model(prepro_image, text)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            # print("Label probs:", ["{0:.10f}".format(i) for i in probs[0]])
            if probs[0][-1] < 0.5:
                detected_objects[np.argmax(probs[0])] = center_loc[i]
        return detected_objects
        

def FindObject(object_name):
    print("Finding an object named {}".format(object_name))
    pass

def PickObject(object_location):
    print("Picking an object located at {}".format(object_location))
    pass

def PlaceObject(object_location):
    print("Placing an object at {}".format(object_location))
    pass

def PourObject(object_location):
    print("Pouring an object at {}".format(object_location))
    pass

def OpenObject(object_location):
    print("Opening an object at {}".format(object_location))
    pass

def PushButton(object_location):
    print("Pushing a button at {}".format(object_location))
    pass

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model and image preprocessing
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    # Set up the image URL
    image_name = "/home/bass/Pictures/tea_packet.jpg"

    # load image
    image = Image.open(image_name)

    # pre-process image
    image = preprocess(image).unsqueeze(0).to(device)
    print("\n\nTensor shape:")
    print(image.shape)
    
    with torch.no_grad():
        image_features = model.encode_image(image)
    print(image_features.shape)
    
    text_snippets = ["a photo of a tea", "a photo of a tea packet", "a photo of a coffee", "a photo of a coffee packet", "a photo of something else"]
    # text_snippets = ["a photo of a dog", "a photo of a cat", "a photo of a bird", "a photo of a fish", "a photo of something else"]

    # pre-process text
    text = clip.tokenize(text_snippets).to(device)
    print(text.shape)
    
    with torch.no_grad():
        text_features = model.encode_text(text)
    print(text_features.shape)
    
    with torch.no_grad():
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    print("Label probs:", ["{0:.10f}".format(i) for i in probs[0]])
    