from os import listdir
import os
import shutil
import re
from os.path import isfile, join
import json

def filter_annotations_for_images():

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

def move_images_tu_subfolders():
    image_filenames = [f for f in listdir('images') if isfile(join('images', f))]

    for image_filename in image_filenames:

        filename_search = re.search('(\d\d).jpg$', image_filename, re.IGNORECASE)
        number  = filename_search.group(1)

        if not os.path.exists(os.path.join('images', '_' + str(number))):
            os.makedirs(os.path.join('images', '_' + str(number)))

        shutil.move(
            os.path.join('images', image_filename),
            os.path.join('images', '_' + str(number), image_filename)
        )
