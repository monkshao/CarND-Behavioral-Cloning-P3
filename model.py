import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
lines = []

#load csv file
# format example
##center,left,right,steering,throttle,brake,speed
##IMG/center_2016_12_01_13_30_48_287.jpg, IMG/left_2016_12_01_13_30_48_287.jpg, IMG/right_2016_12_01_13_30_48_287.jpg, 0, 0, 0, 22.14829
with open('/opt/carnd_p3/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)


#split data into training and validation set
training_data_csv_lines, validation_data_csv_lines = train_test_split(lines, test_size = 0.2)
print("size of input training data:" + str(len(training_data_csv_lines)))
print("size of input validation data:" + str(len(validation_data_csv_lines)))

batch_size = 32
#crop and resize also change color space
def change_color(image):
    #crop_image = image[80:140, :]
    #resize_image_pixel = cv2.resize(crop_image, (32, 32))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

#generate processed images for training and validation data set
def image_generator(samples, batch_size=16):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        data = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            images = []
            angles = []
            if offset+ batch_size < len(data):
                batch_samples = data[offset: offset + batch_size]
                for line in batch_samples:
                    center_img = cv2.imread('/opt/carnd_p3/data/' + line[0].strip())
                    if center_img is not None :
                        # print(center_img.shape)
                        center_img = change_color(center_img)
                        steering_angle = float(line[3].strip())
                        # appending original image
                        images.append(center_img)
                        angles.append(steering_angle)

                        # appending flipped image
                        images.append(np.fliplr(center_img))
                        angles.append(-steering_angle)

                    # appending left camera image
                    left_img = cv2.imread('/opt/carnd_p3/data/' + line[1].strip())
                    if left_img is not None :
                        # print(left_img.shape)
                        left_img = change_color(left_img)
                        images.append(left_img)
                        angles.append(steering_angle + 0.2)

                    # appending right camera image and steering angle with offset
                    right_img = cv2.imread('/opt/carnd_p3/data/' + line[2].strip())
                    if right_img is not None :
                        right_img = change_color(right_img)
                        images.append(right_img)
                        angles.append(steering_angle - 0.2)

                # converting to numpy array
                X_train = np.array(images)
                y_train = np.array(angles)
                #print("length of X_train for this batch:" + len(X_train))
                yield shuffle(X_train, y_train)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


#creating model to be trained
model = Sequential()
# normalize the data, shift the mean to 0

model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(3,160,320)))
kernel_size = (5, 5)
model.add(Convolution2D(24, kernel_size, padding='same', strides=(2, 2), activation = 'elu'))
model.add(Convolution2D(36, kernel_size, padding='same', strides=(2, 2), activation = 'elu'))
model.add(Convolution2D(48, kernel_size, padding='same', strides=(2, 2), activation = 'elu'))
kernel_size = (3, 3)
model.add(Convolution2D(64, kernel_size, padding='same', activation = 'elu'))
model.add(Convolution2D(64, kernel_size, padding='same', activation = 'elu'))
model.add(Dropout(0.5))
#model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#compiling and running the model
model.compile(loss='mse', optimizer='adam')

model.fit_generator(image_generator(training_data_csv_lines, batch_size),
                    steps_per_epoch = int(len(training_data_csv_lines)/batch_size) -1,
                    #nb_epoch = 2,
                    epochs= 5,
                    verbose=1,
                    validation_data=image_generator(validation_data_csv_lines,batch_size),
                    validation_steps=int(len(validation_data_csv_lines)/batch_size) -1 )

#saving the model
model.save('model.h5')