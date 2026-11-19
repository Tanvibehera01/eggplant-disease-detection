from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_dir = r"D:\DOMAIN PROJECT\OUTPUT"

datagen = ImageDataGenerator(
    validation_split=0.2,  # 80% train, 20% validation
    rescale=1./255
)

train_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    subset='training'
)

val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    subset='validation'
)
print("Class mapping:", train_gen.class_indices)