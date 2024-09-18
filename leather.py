import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import keras

def create_cnn_model(inputShape, numClasses, activationNotDense):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation=activationNotDense, input_shape=inputShape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation=activationNotDense))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation=activationNotDense))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation=activationNotDense))
    model.add(layers.Dense(numClasses, activation='softmax'))
    return model

def create_resnet_model(inputShape, numClasses, activationNotDense):
    base_model = tf.keras.applications.ResNet50(input_shape=inputShape, include_top=False, weights='imagenet')
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(numClasses, activation='softmax')
    ])
    return model

def create_vgg_model(inputShape, numClasses, activationNotDense):
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), activation=activationNotDense, input_shape=inputShape))
    model.add(layers.Conv2D(64, (3, 3), activation=activationNotDense))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation=activationNotDense))
    model.add(layers.Conv2D(128, (3, 3), activation=activationNotDense))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(256, (3, 3), activation=activationNotDense))
    model.add(layers.Conv2D(256, (3, 3), activation=activationNotDense))
    model.add(layers.Conv2D(256, (3, 3), activation=activationNotDense))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation=activationNotDense))
    model.add(layers.Dense(numClasses, activation='softmax'))
    return model


dataDirectory = 'F:/Usuarios/Desktop/Freudenberg/machine-learning-defect-leather-py/Leather_Kaggle'
classLabels = ['Folding marks', 'Grain off', 'Growth marks', 'loose grains', 'non defective', 'pinhole']
numClasses = len(classLabels)
inputShape = (224, 224, 3)
activationNotDense = "relu"

trainGen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
testGen = ImageDataGenerator(rescale=1./255)

#Create sets for training validation and testing
trainGenerator = trainGen.flow_from_directory(
    dataDirectory,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    subset='training',
)

validationGenerator = trainGen.flow_from_directory(
    dataDirectory,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
)

testGenerator = testGen.flow_from_directory(
    dataDirectory,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
)

#Create models
cnnModel = create_cnn_model(inputShape, numClasses, activationNotDense)
resnetModel = create_resnet_model(inputShape, numClasses, activationNotDense)
vggModel = create_vgg_model(inputShape, numClasses, activationNotDense)


#Compile models
cnnModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', keras.metrics.CategoricalAccuracy()])
resnetModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', keras.metrics.CategoricalAccuracy()])
vggModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', keras.metrics.CategoricalAccuracy()])

history_cnn = cnnModel.fit(trainGenerator, epochs=10, validation_data=validationGenerator)
history_resnet = resnetModel.fit(trainGenerator, epochs=10, validation_data=validationGenerator)
history_vgg = vggModel.fit(trainGenerator, epochs=10, validation_data=validationGenerator)

plt.plot(history_cnn.history['accuracy'], label='CNN Training Accuracy')
plt.plot(history_resnet.history['accuracy'], label='ResNet Training Accuracy')
plt.plot(history_vgg.history['accuracy'], label='VGG Training Accuracy')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Comparison - Training Accuracy')
plt.legend()
plt.show()