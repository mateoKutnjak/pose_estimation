from os import listdir
from os.path import isfile, join
import json

image_filenames = [f for f in listdir('_images') if isfile(join('_images', f))]

result = []

with open('annotations/annotations.json') as f:
    json_parsed = json.loads(f.read())

    for image_filename in image_filenames:
        for annotations_data in json_parsed:
            if annotations_data['img_paths'] == image_filename:
                result.append(annotations_data)

with open('_annotations/annotations.json', 'a+') as f:
    f.write(json.dumps(result))