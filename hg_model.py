import hg_layers
from dataset import MPII_dataset


class HourglassModel:

    def __init__(self, images_path, annotations_path, channels, classes, stacks, input_shape, output_shape):
        self.images_path = images_path
        self.annotations_path = annotations_path
        self.channels = channels
        self.classes = classes
        self.stacks = stacks
        self.input_shape = input_shape
        self.output_shape = output_shape

    def build(self):
        self.model = hg_layers.create_network(
            self.input_shape,
            self.channels,
            self.classes,
            self.stacks
        )

    def train(self):
        dataset = MPII_dataset(self.annotations_path)
