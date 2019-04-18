import os
from keras.callbacks import Callback
from dataset import MPII_dataset
import eval_utils


class EvalCallback(Callback):

    def __init__(self, save_dir, batch_size, stacks_num, input_shape, output_shape):
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.stacks_num = stacks_num
        self.input_shape = input_shape
        self.output_shape = output_shape


    def on_epoch_end(self, epoch, logs={}):
        # if epoch == 0:
        #     jsonfile = os.path.join(self.save_dir, "architecture.json")
        #     with open(jsonfile, "w") as f:
        #         f.write(self.model.as_json())
        #
        # model_path = os.path.join(self.save_dir, "weights_" + str(epoch) + ".h5")
        # self.model.save_weights(model_path)

        accuracy = self.epoch_evaluation(epoch)

        with open(os.path.join(self.save_dir, 'epoch_validations.txt'), 'a+') as f:
            f.write('Epoch ' + str(epoch) + ' with accuracy of ' + str(accuracy) + '\n')


    def epoch_evaluation(self, epoch):
        mpii_valid_dataset = MPII_dataset(
            images_dir='_images',
            annots_json_filename='_annotations/annotations.json',
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            type='valid'
        )

        total_correct, total_incorrect = 0.0, 0.0
        examples_count = 0

        for input_batch, output_batch, metadata_batch in mpii_valid_dataset.generate_batches(
                batch_size=self.batch_size,
                stacks_num=self.stacks_num,
                metadata_flag=True
        ):

            print(examples_count, self.batch_size, mpii_valid_dataset.get_dataset_size())
            examples_count += self.batch_size
            if examples_count > mpii_valid_dataset.get_dataset_size():
                break



            prediction_batch = self.model.predict(input_batch)

            corrects, incorrects = eval_utils.heatmap_batch_accuracy(
                prediction_batch=prediction_batch[-1],
                metadata_batch=metadata_batch,
                threshold=0.5
            )

            total_correct += corrects
            total_incorrect += incorrects

        return total_correct / (total_correct + total_incorrect)