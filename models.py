import layers
from eval_callback import EvalCallback


class HourglassModel:

    def __init__(self, dataset, images_dir, annotations_json_file, log_dir, batch_size, channels_num=256, classes_num=16, stacks_num=8):
        self.dataset = dataset
        self.images_dir = images_dir
        self.annotations_json_file = annotations_json_file
        self.log_dir = log_dir
        self.batch_size = batch_size
        self.channels_num = channels_num
        self.classes_num = classes_num
        self.stacks_num = stacks_num

        self.model = None

    def build(self):
        self.model = layers.create_network(
            input_shape=self.dataset.get_input_shape(),
            batch_size=self.batch_size,
            channels=self.channels_num,
            classes=self.classes_num,
            stacks=self.stacks_num
        )

        # self.model.summary()

    def train(self, epochs):
        train_generator = self.dataset.generate_batches(
            batch_size=self.batch_size,
            stacks_num=self.stacks_num,
        )

        checkpoint = EvalCallback(
            images_dir=self.images_dir,
            annotations_json_file=self.annotations_json_file,
            log_dir=self.log_dir,
            batch_size=self.batch_size,
            stacks_num=self.stacks_num,
            input_shape=self.dataset.get_input_shape(),
            output_shape=self.dataset.get_output_shape()
        )

        self.model.fit_generator(
            generator=train_generator,
            steps_per_epoch=self.dataset.get_dataset_size() // self.batch_size,
            epochs=epochs,
            callbacks=[checkpoint]
        )
