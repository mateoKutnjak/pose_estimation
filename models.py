import layers
import math
import os
import datetime

from dataset import MPII_dataset
from eval_callback import EvalCallback

from tensorflow.python.keras.callbacks import CSVLogger
from tensorflow.python.keras.models import load_model


class HourglassModel:

    def __init__(self, images_dir, annotations_json_file, log_dir, batch_size, input_shape, output_shape, channels_num=256, classes_num=16, stacks_num=8):
        self.images_dir = images_dir
        self.annotations_json_file = annotations_json_file
        self.log_dir = log_dir
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.channels_num = channels_num
        self.classes_num = classes_num
        self.stacks_num = stacks_num

        self.model = None

    def build(self):
        print('Building model... ')
        self.model = layers.create_network(
            input_shape=self.input_shape,
            batch_size=self.batch_size,
            channels=self.channels_num,
            classes=self.classes_num,
            stacks=self.stacks_num
        )

        # self.model.summary()
        print('DONE')

    def train(self, epochs):
        train_dataset = MPII_dataset(
            images_dir=self.images_dir,
            annots_json_filename=self.annotations_json_file,
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            type='train'
        )

        train_generator = train_dataset.generate_batches(
            batch_size=self.batch_size,
            stacks_num=self.stacks_num,
        )

        checkpoint = EvalCallback(
            images_dir=self.images_dir,
            annotations_json_file=self.annotations_json_file,
            log_dir=self.log_dir,
            batch_size=self.batch_size,
            stacks_num=self.stacks_num,
            input_shape=train_dataset.get_input_shape(),
            output_shape=train_dataset.get_output_shape()
        )

        logger_filepath = os.path.join(self.log_dir, "csv_train.csv")

        if not os.path.exists(logger_filepath):
            open(logger_filepath, "w").close()

        logger = CSVLogger(logger_filepath, append=True)

        self.model.fit_generator(
            generator=train_generator,
            steps_per_epoch=int(math.ceil(train_dataset.get_dataset_size() // self.batch_size)),
            epochs=epochs,
            callbacks=[checkpoint, logger]
        )

    def resume_train(self, epochs, checkpoint_epoch):
        self.model = load_model(os.path.join(self.log_dir, 'model_architecture.h5'))
        self.model.load_weights(os.path.join(self.log_dir, 'model_weights.h5'))

        self.model = layers.compile_model(model=self.model)

        train_dataset = MPII_dataset(
            images_dir=self.images_dir,
            annots_json_filename=self.annotations_json_file,
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            type='train'
        )

        train_generator = train_dataset.generate_batches(
            batch_size=self.batch_size,
            stacks_num=self.stacks_num,
        )

        checkpoint = EvalCallback(
            images_dir=self.images_dir,
            annotations_json_file=self.annotations_json_file,
            log_dir=self.log_dir,
            batch_size=self.batch_size,
            stacks_num=self.stacks_num,
            input_shape=train_dataset.get_input_shape(),
            output_shape=train_dataset.get_output_shape()
        )

        logger = CSVLogger(
            os.path.join(self.log_dir, "csv_train.csv"),
            append=True
        )

        self.model.fit_generator(
            generator=train_generator,
            steps_per_epoch=int(math.ceil(train_dataset.get_dataset_size() // self.batch_size)),
            epochs=epochs,
            callbacks=[checkpoint, logger],
            initial_epoch=checkpoint_epoch
        )