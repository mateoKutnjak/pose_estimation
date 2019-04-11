from dataset import MPII_dataset
from models import HourglassModel


IMAGE_DIR = 'images/'
ANNOTATION_JSON_FILENAME = 'annotations/mpii_annotations.json'
INPUT_RESOLUTION = (256, 256, 3)
OUTPUT_RESOLUTION = (64, 64, 16)
BATCH_SIZE = 4


mpii_dataset = MPII_dataset(
    images_dir=IMAGE_DIR,
    annots_json_filename=ANNOTATION_JSON_FILENAME,
    input_shape=INPUT_RESOLUTION,
    output_shape=OUTPUT_RESOLUTION
)

input_batch, output_batch = mpii_dataset.create_batches(batch_size=BATCH_SIZE)

import pdb
pdb.set_trace()

hg_model = HourglassModel(dataset=mpii_dataset, batch_size=BATCH_SIZE)
