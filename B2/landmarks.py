import os
import numpy as np
from keras_preprocessing import image
import cv2
import dlib
import tensorflow as tf

# PATH TO ALL IMAGES
global basedir, image_paths, target_size, images_dir
basedir = "Datasets/"
images_dir = os.path.join(basedir, "cartoon_set/img/")
labels_filename = "cartoon_set/labels.csv"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def get_data():
    images_dir = os.path.join(basedir, "cartoon_set/img/")
    labels_filename = "cartoon_set/labels.csv"
    X, y = extract_features_labels(images_dir, labels_filename)
    Y = np.array([y, -(y - 1)]).T
    tr_X = X[:10000]
    tr_Y = Y[:10000]

    images_dir = os.path.join(basedir, "cartoon_set_test/img/")
    labels_filename = "cartoon_set_test/labels.csv"
    X, y = extract_features_labels(images_dir, labels_filename)
    Y = np.array([y, -(y - 1)]).T
    te_X = X[:2000]
    te_Y = Y[:2000]

    return tr_X, tr_Y, te_X, te_Y


def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def run_dlib_shape(image):
    # in this function we load the image, detect the landmarks of the face, and then return the image and the landmarks

    resized_image = image.astype("uint8")

    img = resized_image[:, :, :3]

    # detect faces in the grayscale image
    rects = detector(img, 1)
    num_faces = len(rects)

    if num_faces == 0:
        return None, image

    eye_colors = []
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        temp_shape = predictor(resized_image, rect)

        temp_shape = shape_to_np(temp_shape)

        # extract the left and right eye points from the landmark points
        left_eye = temp_shape[36:42]
        right_eye = temp_shape[42:48]

        # extract the left and right eye regions
        left_eye_region = img[
            left_eye[1][1] : left_eye[5][1], left_eye[0][0] : left_eye[3][0]
        ]
        right_eye_region = img[
            right_eye[1][1] : right_eye[5][1], right_eye[0][0] : right_eye[3][0]
        ]

        # calculate the average RGB values of the left and right eye regions
        left_eye_color = np.mean(left_eye_region, axis=(0, 1))
        right_eye_color = np.mean(right_eye_region, axis=(0, 1))

        # append the average RGB values to the list
        eye_colors.append(left_eye_color)
        eye_colors.append(right_eye_color)

    return eye_colors, resized_image


def extract_features_labels(images_dir, labels_filename):
    """
    This funtion extracts the landmarks features for all images in the folder 'dataset/celeba'.
    It also extracts the gender label for each image.
    :return:
        landmark_features:  an array containing 68 landmark points for each image in which a face was detected
        gender_labels:      an array containing the gender label (male=0 and female=1) for each image in
                            which a face was detected
    """
    print("Starting feature extraction...")
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    target_size = None
    labels_file = open(os.path.join(basedir, labels_filename), "r")
    lines = labels_file.readlines()
    # print(lines)
    # print(lines[1].split('\t')[3])

    face_labels = {line.split("\t")[0]: int(line.split("\t")[1]) for line in lines[1:]}

    if os.path.isdir(images_dir):
        all_features = []
        all_labels = []

        for img_path in image_paths:
            file_name = img_path.split("/")[-1].split(".")[-2]

            # load image
            img = image.img_to_array(
                image.load_img(
                    img_path, target_size=target_size, interpolation="bicubic"
                )
            )
            features, _ = run_dlib_shape(img)
            if features is not None:
                all_features.append(features)

                all_labels.append(face_labels[file_name])
                print(file_name)

    landmark_features = np.array(all_features)
    face_labels = np.array(
        all_labels
    )  # simply converts the -1 into 0, so male=0 and female=1
    return landmark_features, face_labels
