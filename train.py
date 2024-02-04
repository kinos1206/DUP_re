import time
from options.train_options import TrainOptions
from data.custom_dataset_dataloader import CreateDataLoader
from model.pixelization_model import PixelizationModel
import os
from PIL import Image
import torch
import torch._dynamo
from matplotlib import pyplot as plt

torch._dynamo.config.suppress_errors = True
torch.autograd.set_detect_anomaly(True)

opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = PixelizationModel()
model.initialize(opt)
total_steps = 0
grid_loss_list, pixel_loss_list, depixel_loss_list, grid_D_list, pixel_D_list, depixel_D_list = [], [], [], [], [], []  

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy.astype('uint8'))
    image_pil.save(image_path)

def print_current_errors(epoch, i, errors, t, t_data, log_name):
    message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, i, t, t_data)
    for k, v in errors.items():
        message += '%s: %.3f ' % (k, v)

    print(message)
    with open(log_name, "a") as log_file:
        log_file.write('%s\n' % message)
                      
for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    iter_data_time = time.time()
    epoch_iter = 0

    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        if total_steps % opt.print_freq == 0:
            t_data = iter_start_time - iter_data_time
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(data)
        model.optimize_parameters()

        if total_steps % opt.print_freq == 0:
            img_dir = os.path.join(opt.checkpoints_dir, 'images')
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)
            for label, image_numpy in model.get_current_visuals_train().items():
                img_path = os.path.join(img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                save_image(image_numpy, img_path)

            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            log_name = os.path.join(opt.checkpoints_dir, 'loss_log.txt')
            print_current_errors(epoch, epoch_iter, errors, t, t_data, log_name)
            
        iter_data_time = time.time()
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    model.update_learning_rate()

    grid_loss_list.append(model.loss_gridnet_item)
    pixel_loss_list.append(model.loss_pixelnet_item)
    depixel_loss_list.append(model.loss_depixelnet_item)
    grid_D_list.append(model.loss_D_gridnet)
    pixel_D_list.append(model.loss_D_pixelnet)
    depixel_D_list.append(model.loss_D_depixelnet)

# plot graph
fig = plt.figure()
plt.plot(range(opt.niter + opt.niter_decay), grid_loss_list, color='blue', linestyle='-', label='grid_loss')
plt.plot(range(opt.niter + opt.niter_decay), pixel_loss_list, color='green', linestyle='-', label='pixel_loss')
plt.plot(range(opt.niter + opt.niter_decay), depixel_loss_list, color='red', linestyle='-', label='depixel_loss')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Training loss')
plt.grid()
fig2 = plt.figure()
plt.plot(range(opt.niter + opt.niter_decay), grid_D_list, color='blue', linestyle='-', label='grid_D_loss')
plt.plot(range(opt.niter + opt.niter_decay), pixel_D_list, color='green', linestyle='-', label='pixel_D_loss')
plt.plot(range(opt.niter + opt.niter_decay), depixel_D_list, color='red', linestyle='-', label='depixel_D_loss')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('D_loss')
plt.ylim(0,3)
plt.title('Training D_loss')
plt.grid()
dirname1 = "logs/loss/"
dirname2 = "logs/D/"
os.makedirs(dirname1, exist_ok=True)
os.makedirs(dirname2, exist_ok=True)
fig.savefig(dirname1 + "loss_data%d_rate%f_niter%d_niterdecay%d.png" % (dataset_size, opt.lr, opt.niter, opt.niter_decay))
fig2.savefig(dirname2 + "D_data%d_rate%f_niter%d_niterdecay%d.png" % (dataset_size, opt.lr, opt.niter, opt.niter_decay))