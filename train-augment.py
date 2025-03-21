# This code is released under the CC BY-SA 4.0 license.

import torch
print(torch.cuda.is_available())

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Check if the environment variable is set
if os.getenv('KMP_DUPLICATE_LIB_OK') == 'TRUE':
    print("Environment variable set successfully.")
else:
    print("Failed to set environment variable.")
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
# from util.visualizer import Visualizer

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    opt.dataroot = 'dataset.pickle'  # 使用我们生成的数据集
    opt.dataset_mode = 'patch'     # 使用新的patch数据集模式

    opt.continue_train = False
    opt.name = 'exp_1'
    opt.model = 'da_cytran'
    # opt.load_iter = 4
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.

    print(f"数据集大小: {dataset_size}")
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # # visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>

        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        # visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration

            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                # model.compute_visuals()
                
                # visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, epoch_iter, t_comp, t_data)
                for k, v in losses.items():
                    message += '%s: %.3f ' % (k, v)
                print(message)  # print the message

                if not os.path.exists(os.path.join(opt.checkpoints_dir, opt.name)):
                    os.makedirs(os.path.join(opt.checkpoints_dir, opt.name))

                log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
                with open(log_name, "a") as log_file:                                                                                                                                                                                                
                    log_file.write('%s\n' % message)  # save the message

                # visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                # if opt.display_id > 0:
                #     visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        model.update_learning_rate()    # update learning rates in the beginning of every epoch.

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
