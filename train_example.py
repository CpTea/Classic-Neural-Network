import tensorflow as tf
from tensorflow import keras
from resnet import resnet18, resnet50


# ----------------------------------------------------------------------------------
model = resnet18(num_classes=2)
# ----------------------------------------------------------------------------------
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
# ----------------------------------------------------------------------------------
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)
# ----------------------------------------------------------------------------------
from make_dataset import Dataset
IMG_WIDTH  = 224
IMG_HEIGHT = 224
BATCH_SIZE = 2

train_ds = Dataset('./dat/train', IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE, loss_func='SparseCategoricalCrossentropy').data
test_ds  = Dataset('./dat/test' , IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE, loss_func='SparseCategoricalCrossentropy').data
# ----------------------------------------------------------------------------------

EPOCHS = 100
for epoch in range(EPOCHS):
    for images, labels in train_ds:
        train_step(images, labels)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    template = 'Epoch {}, Loss: {:.4f}, Accuracy: {:.2f}, Test Loss: {:.4f}, Test Accuracy: {:.2f}'
    print (template.format( epoch + 1,
                            train_loss.result(),
                            train_accuracy.result()*100,
                            test_loss.result(),
                            test_accuracy.result()*100))