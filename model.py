import csv
import cv2
import numpy as np
import os.path
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
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

lap2_lines=[]
## load self uploaded-images for lap2
with open('/home/workspace/lap2_images/lap2_videos/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lap2_lines.append(line)


#split data into training and validation set
training_data_csv_lines_lap2, validation_data_csv_lines_lap2 = train_test_split(lap2_lines, test_size = 0.2)
print("size of input training data for lap2:" + str(len(training_data_csv_lines_lap2)))
print("size of input validation data for lap2:" + str(len(validation_data_csv_lines_lap2)))

merged_training_data_csv_lines= training_data_csv_lines + training_data_csv_lines_lap2
merged_validation_data_csv_lines = validation_data_csv_lines + validation_data_csv_lines_lap2


print("size of total input training data for lap1 + lap2:" + str(len(merged_training_data_csv_lines)))
print("size of total input validation data for lap1 + lap2:" + str(len(merged_validation_data_csv_lines)))

batchSize = 64
#crop and resize also change color space
def change_color(image):
    #crop_image = image[80:140, :]
    #resize_image_pixel = cv2.resize(crop_image, (32, 32))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

#generate processed images for training and validation data set
def image_generator(samples, data1_dir, data2_dir, batch_size=16):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        data = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            images = []
            angles = []
            center_img = None
            left_img = None
            right_img = None
            if offset+ batch_size < len(data):
                batch_samples = data[offset: offset + batch_size]
                for line in batch_samples:
                    if os.path.isfile(data1_dir + line[0].strip()) :
                        center_img = cv2.imread(data1_dir + line[0].strip())
                    elif os.path.isfile(data2_dir+'/IMG/' + line[0].strip().split('/')[-1]) :
                        center_img = cv2.imread(data2_dir+'/IMG/' + line[0].strip().split('/')[-1])
                    if center_img is not None :
                        steering_angle = None
                        try:
                            steering_angle = float(line[3].strip())
                        except ValueError:
                            print(line[3].strip() + " is Not a float")
                            continue
                        # print(center_img.shape)
                        center_img = change_color(center_img)
                        # appending original image
                        images.append(center_img)
                        angles.append(steering_angle)

                        # appending flipped image
                        images.append(np.fliplr(center_img))
                        angles.append(-steering_angle)

                    # appending left camera image
                    if os.path.isfile(data1_dir + line[1].strip()) :
                        left_img = cv2.imread(data1_dir + line[1].strip())
                    elif os.path.isfile(data2_dir+'/IMG/' + line[1].strip().split('/')[-1]) :
                        left_img = cv2.imread(data2_dir+'/IMG/' + line[0].strip().split('/')[-1])
                    if left_img is not None :
                        # print(left_img.shape)
                        left_img = change_color(left_img)
                        images.append(left_img)
                        angles.append(steering_angle + 0.2)

                    # appending right camera image and steering angle with offset
                    if os.path.isfile(data1_dir + line[2].strip()) :
                        right_img = cv2.imread(data1_dir + line[2].strip())
                    elif os.path.isfile(data2_dir+'/IMG/' + line[2].strip().split('/')[-1]) :
                        right_img = cv2.imread(data2_dir+'/IMG/' + line[0].strip().split('/')[-1])
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
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#compiling and running the model
model.compile(loss='mse', optimizer='adam')

# Train Keras model, saving the model whenever improvements are made and stopping if loss does not improve.
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0, patience=2)
checkpointer = ModelCheckpoint(filepath='./model-{val_loss:.5f}.h5', verbose=1, save_best_only=True)

model.fit_generator(image_generator(merged_training_data_csv_lines,
                                    data1_dir='/opt/carnd_p3/data/',
                                    data2_dir='/home/workspace/lap2_images/lap2_videos/',
                                    batch_size = batchSize),
                    steps_per_epoch = int(len(merged_training_data_csv_lines)/batchSize) -1,
                    #nb_epoch = 2,
                    epochs= 5,
                    verbose=1,
                    validation_data=image_generator(merged_validation_data_csv_lines,
                                                    data1_dir='/opt/carnd_p3/data/',
                                                    data2_dir='/home/workspace/lap2_images/lap2_videos/',
                                                    batch_size = batchSize),
                    validation_steps=int(len(merged_validation_data_csv_lines)/batchSize) -1,
                    callbacks=[early_stopping, checkpointer])

#saving the model
model.save('model-2.h5')