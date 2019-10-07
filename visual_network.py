import tensorflow as tf 
from tensorflow import keras


def networkRecorder(logdir, model, input_shape):
    @tf.function
    def forward(model, x): return model(x)
    x = tf.zeros(input_shape)
    writer = tf.summary.create_file_writer(logdir)
    tf.summary.trace_on(graph=True, profiler=True)
    forward(model, x)
    with writer.as_default():
        tf.summary.trace_export(name='model_trace', step=0,  profiler_outdir=logdir)


if __name__ == '__main__':
    from datetime import datetime
    from resnet import resnet50
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = '.\logs\%s' % stamp
    shape = [1, 224, 224, 3]
    networkRecorder(logdir, resnet50(), shape)
