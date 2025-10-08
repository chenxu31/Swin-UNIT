import time
import datetime
import torch
from options.train_options import TrainOptions
#from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import platform
import numpy
import sys


if platform.system() == 'Windows':
    NUM_WORKERS = 0
    UTIL_DIR = r"E:\我的坚果云\sourcecode\python\util"
else:
    NUM_WORKERS = 4
    UTIL_DIR = r"/home/chenxu/我的坚果云/sourcecode/python/util"

sys.path.append(UTIL_DIR)
import common_net_pt as common_net
import common_metrics
import common_pelvic_pt as common_pelvic
import common_cmf_pt as common_cmf


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    #dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    #dataset_size = len(dataset)    # get the number of images in the dataset.

    if opt.task == "pelvic":
        common_file = common_pelvic
        dataset_s = common_pelvic.Dataset(opt.dataroot, "ct", n_slices=opt.input_nc, debug=opt.debug)
        dataset_t = common_pelvic.Dataset(opt.dataroot, "cbct", n_slices=opt.input_nc, debug=opt.debug)
        val_data_s, val_data_t, _, _ = common_pelvic.load_val_data(opt.dataroot, valid=True)
    elif opt.task == "cmf":
        common_file = common_amos
        dataset_s = common_cmf.Dataset(opt.dataroot, modality="ct", n_slices=opt.input_nc, debug=opt.debug)
        dataset_t = common_cmf.Dataset(opt.dataroot, modality="mr", n_slices=opt.input_nc, debug=opt.debug)
        val_data_t, val_data_s, _ = common_cmf.load_test_data(opt.dataroot)
    else:
        assert 0

    if opt.debug:
        val_data_s = val_data_s[:1]
        val_data_t = val_data_t[:1]

    patch_shape = (opt.input_nc, dataset_s.patch_height, dataset_s.patch_width)
    dataloader_s = torch.utils.data.DataLoader(dataset_s, batch_size=opt.batch_size, shuffle=True, pin_memory=True, drop_last=True)
    dataloader_t = torch.utils.data.DataLoader(dataset_t, batch_size=opt.batch_size, shuffle=True, pin_memory=True, drop_last=True)


    model = create_model(opt)      # create a model given opt.model and other options
    #print('The length of training datasets = %d' % dataset_size)
    time_str = str(datetime.datetime.now().strftime('_%m%d%H%M'))
    opt.display_env = opt.name + time_str
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    opt.visualizer = visualizer
    total_iters = 0                # the total number of training iterations

    optimize_time = 0.1

    times = []
    best_psnr = 0
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        # visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        #dataset.set_epoch(epoch)
        # print("len_dataset:", len(dataset)) #76
        #for i, data in enumerate(dataset):  # inner loop within one epoch
        for i,(data_s, data_t) in enumerate(zip(dataloader_s, dataloader_t)):
            #iter_start_time = time.time()  # timer for computation per iteration
            #if total_iters % opt.print_freq == 0:
            #    t_data = iter_start_time - iter_data_time
            data = {
                "A": data_t["image"],
                "B": data_s["image"],
            }

            #batch_size = opt.batch_size
            #total_iters += batch_size
            #epoch_iter += batch_size
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            #optimize_start_time = time.time()
            if epoch == opt.epoch_count and i == 0:
                model.data_dependent_initialize(data)
                model.setup(opt)               # regular setup: load and print networks; create schedulers
                model.parallelize()
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()

            """
            optimize_time = (time.time() - optimize_start_time) / batch_size * 0.005 + 0.995 * optimize_time

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                evals = model.get_current_evals()
                visualizer.print_current_losses(epoch, epoch_iter, losses, optimize_time, t_data)
                if opt.display_id is None or opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, evals)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                print(opt.name)  # it's useful to occasionally show the experiment name on console
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)
        """

        #print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.

        model.netG_A.eval()
        val_psnr = numpy.zeros((val_data_s.shape[0], ), numpy.float32)
        with torch.no_grad():
            for i in range(len(val_data_t)):
                syn = common_net.produce_results(next(model.netG_A.parameters()).device,
                                                 lambda x: model.netG_A(x)[0], [patch_shape, ], [val_data_t[i], ],
                                                 data_shape=val_data_t[i].shape, patch_shape=patch_shape, is_seg=False)
                val_psnr[i] = common_metrics.psnr(syn, val_data_s[i])

        model.netG_A.train()
        if val_psnr.mean() > best_psnr:
            best_psnr = val_psnr.mean()
            model.save_networks('best')
        print('Epoch %d  val_psnr: %.4f  best_psnr: %.4f' % (epoch, val_psnr.mean(), best_psnr))

        model.save_networks('last')