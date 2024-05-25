import cv2
import numpy as np
import os
import pyttsx3

path_to_folder = 'ImageQuery'
orb_detector = cv2.ORB_create(nfeatures=500)

#### Import Images
image_list = []
class_name_list = []
folder_list = os.listdir(path_to_folder)
print('Total Classes Detected', len(folder_list))
for folder_name in folder_list:
    if folder_name == '.DS_Store':
        continue  # Skip this file
    current_image = cv2.imread(f'{path_to_folder}/{folder_name}', 0)
    image_list.append(current_image)
    class_name_list.append(os.path.splitext(folder_name)[0])
print(class_name_list)

# Function to speak the given text
def speak_text(text_to_speak):
    os.system(f'say {text_to_speak}')

def find_descriptors(images):
    descriptor_list = []
    for image in images:
        keypoints, descriptors = orb_detector.detectAndCompute(image, None)
        descriptor_list.append(descriptors)
    return descriptor_list

def find_matching_id(query_image, descriptor_list, threshold=20):
    query_keypoints, query_descriptors = orb_detector.detectAndCompute(query_image, None)
    if query_descriptors is None:
        print("No descriptors found for query image.")
        return
    
    bf_matcher = cv2.BFMatcher()
    match_scores = []
    final_id = -1
    try:
        for descriptors in descriptor_list:
            if descriptors is None:
                print("Descriptors not found for a reference image.")
                match_scores.append(0)
                continue
            
            matches = bf_matcher.knnMatch(descriptors, query_descriptors, k=2)
            good_matches = []
            for match1, match2 in matches:
                if match1.distance < 0.75 * match2.distance:
                    good_matches.append([match1])
            match_scores.append(len(good_matches))
    except:
        pass
    
    if len(match_scores) != 0:
        if max(match_scores) > threshold:
            final_id = match_scores.index(max(match_scores))
    return final_id

descriptor_list = find_descriptors(image_list)
print(len(descriptor_list))

capture = cv2.VideoCapture(0)

while True:
    success, query_image = capture.read()
    original_image = query_image.copy()
    query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
    
    # Flip the image horizontally
    original_image = cv2.flip(original_image, 1)

    matching_id = find_matching_id(query_image, descriptor_list)
    if matching_id is not None and matching_id != -1:  # Check if matching_id is not None and not -1
        text_to_speak = class_name_list[matching_id]  # Get the text to speak
        cv2.putText(original_image, text_to_speak, (103, 347), cv2.FONT_HERSHEY_COMPLEX, 5, (0, 0, 128), 1)
        speak_text(text_to_speak)  # Speak the text

    cv2.imshow('query_image', original_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

capture.release()
cv2.destroyAllWindows()
