import os
from tensorflow.python.keras.callbacks import Callback
from dataset import MPII_dataset
import eval_utils
import plot_utils


class EvalCallback(Callback):

    def __init__(self, images_dir, annotations_json_file, log_dir, batch_size, stacks_num, input_shape, output_shape):
        self.images_dir = images_dir
        self.annotations_json_file = annotations_json_file
        self.log_dir = log_dir
        self.batch_size = batch_size
        self.stacks_num = stacks_num
        self.input_shape = input_shape
        self.output_shape = output_shape

    # def on_epoch_begin(self, epoch, logs=None):
    #     print('Saving model architecture... ')
    #     self.model.save(os.path.join(self.log_dir, "model_architecture.h5"))
    #     print('DONE')
    #
    #     print('Saving model weights... ')
    #     self.model.save_weights(os.path.join(self.log_dir, "model_weights.h5"))
    #     print('DONE')

    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0:
            print('Saving model architecture... ')
            self.model.save(os.path.join(self.log_dir, "model_architecture.h5"))
            print('DONE')

        if epoch % 200 == 0:

            print('Saving model weights... ')
            self.model.save_weights(os.path.join(self.log_dir, "model_weights.h5"))
            print('DONE')

        accuracy = self.epoch_evaluation(epoch)

        with open(os.path.join(self.log_dir, 'epoch_validations.txt'), 'a+') as f:
            f.write('Epoch ' + str(epoch) + ' with accuracy of ' + str(accuracy) + '\n')

        print('Epoch ended, calculating validation accuracy and writing logs... ')
        print('DONE')


    def epoch_evaluation(self, epoch):
        mpii_valid_dataset = MPII_dataset(
            images_dir=self.images_dir,
            annots_json_filename=self.annotations_json_file,
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            type='valid',
            sigma=2,
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

            # if examples_count % (self.batch_size * 1) == 0: # TODO change
            #     import pdb
            #     pdb.set_trace()
            #     final_heatmaps = prediction_batch[-1]
            #
            #     # for batch_index in range(self.batch_size):
            #     predicted_joints = eval_utils.find_heatmap_joints(final_heatmaps[-1], threshold=0.5)
            #
            #     plot_utils.plot_predicted_joints(
            #         image=input_batch[-1],
            #         obj_joints=predicted_joints,
            #         output_shape=self.output_shape,
            #         save_filename=os.path.join(self.log_dir, str(epoch) + '_' + str(examples_count))
            #     )

            corrects, incorrects = eval_utils.heatmap_batch_accuracy(
                prediction_batch=prediction_batch[-1],
                metadata_batch=metadata_batch,
                threshold=0.5
            )

            total_correct += corrects
            total_incorrect += incorrects

        if total_correct + total_incorrect == 0.0:
            return 0.0
        return total_correct / (total_correct + total_incorrect)