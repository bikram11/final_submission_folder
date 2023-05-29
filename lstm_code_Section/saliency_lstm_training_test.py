import os
import argparse
import time
import math
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F
import torchvision
import numbers
from lstm_model import ConvLSTMNet,ConvLSTMAttentionNet
from bdd_dataloader import BDD_Dataloader


from sklearn.metrics import roc_auc_score



parser = argparse.ArgumentParser(description='Saliency prediction LSTM')

parser.add_argument('--data', metavar='DIR', help='path to extracted features')

parser.add_argument('--best', default='', type=str, metavar='PATH', help='path to best checkpoint used for saving the model and using in test when no-train')

parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='no. of epochs')

parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='learning rate', dest='lr')

parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='initial weight decay (default: 1e-4)',
                    dest='weight_decay')

parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use. if any else cpu')

parser.add_argument('--no_train', action='store_true', help="used only for test", default=False)

parser.add_argument('--gazemaps', metavar='DIR', help='path to folder with gaze map images')
parser.add_argument('--traingrid', default='', type=str, metavar='PATH', help='path to txt with gaze map grid entries for training images')
parser.add_argument('--valgrid', default='', type=str, metavar='PATH', help='path to txt with gaze map grid entries for validation images')
parser.add_argument('--testgrid', default='', type=str, metavar='PATH', help='path to txt with gaze map grid entries for test images')
parser.add_argument('--yolo5bb', metavar='DIR', help='path to folder of yolo5 bounding box txt files comes from the detect file after completing the run')
parser.add_argument('--visualizations', metavar='DIR', help='destination folder')
parser.add_argument('--threshhold', default=0.5, type=float, metavar='N', help='object detection confidence threshold')



def main():
    args = parser.parse_args()

    dim = 256
    th = 1 / dim

    if args.gpu is not None:
        print(f"Use GPU: {args.gpu} for training")

    model = ConvLSTMAttentionNet(16,16, args.sequence)

    

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)

    criterion = nn.BCEWithLogitsLoss().cuda(args.gpu)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08,
                                 weight_decay=args.weight_decay)

    train_loader, val_loader, test_loader = None, None, None
    if not args.no_train:
        train_dataset = BDD_Dataloader("training", args.traingrid, os.path.join(args.data, 'training'), th, args.gazemaps, args.sequence)
        val_dataset = BDD_Dataloader("validation", args.valgrid, os.path.join(args.data, 'validation'), th, args.gazemaps, args.sequence)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,
                                                   num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False,
                                                 num_workers=4, pin_memory=True)

    test_dataset = BDD_Dataloader("test", args.testgrid, os.path.join(args.data, 'test'), th, args.gazemaps, args.sequence)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False,
                                              num_workers=4, pin_memory=True)

    best_loss = 1000000

    if not args.no_train:
        training_losses_for_each_epoch = []
        validation_losses_for_each_epoch = []
        for epoch in range(0, args.epochs):
            adjust_learning_rate(optimizer, epoch, args)

            loss_for_this_epoch = train(train_loader, model, criterion, optimizer, epoch, args)
            training_losses_for_each_epoch.append(loss_for_this_epoch.avg)

            loss1 = validate(val_loader, model, criterion, args)
            validation_losses_for_each_epoch.append(loss1[1].avg)

            is_best = loss1[0] < best_loss
            best_loss = min(loss1[0], best_loss)
            print("\nBest loss: ", best_loss, "Loss from this epoch: ", loss1[0])

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
            }, is_best, args.best)

        print("Training Losses for each epoch: ", training_losses_for_each_epoch)
        print("Validation Losses for each epoch: ", validation_losses_for_each_epoch)

    if args.best:
        if os.path.isfile(args.best):
            print("=> loading checkpoint '{}'".format(args.best))
            checkpoint = torch.load(args.best)
            model.load_state_dict(checkpoint['state_dict'], False)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.best, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    test(test_loader, model, criterion, args)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    if is_best:
        torch.save(state, filename)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    loss_mat = []

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)

        loss = criterion(output, target)
        losses.update(loss.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))
            loss_mat.append(loss.item())

    print(f"Losses:  {loss_mat}")
    return losses


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)

            loss = criterion(output, target)
            losses.update(loss.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                print('Validation: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      .format(
                       i, len(val_loader), batch_time=batch_time, loss=losses))

    return [loss, losses]

def object_metrics(filename, heatmap_img, gt_img, gt, hm_max_values, args):
    tp = 0
    fp = 0
    fn = 0
    all_count = 0

    if os.path.exists(filename):
        with open(filename) as f:
            for linestring in f:
                all_count += 1
                line = linestring.split()

                width = float(line[3])
                height = float(line[4])
                x_center = float(line[1])
                y_center = float(line[2])

                x_min, x_max, y_min, y_max = compute_absolute_bounding_box(x_center, y_center, width, height)

                # find maximum pixel value within object bounding box
                gt_obj = gt_img[0, y_min:y_max+1, x_min:x_max+1]
                gt_obj_max = torch.max(gt_obj)
                heatmap_obj = heatmap_img[0, y_min:y_max+1, x_min:x_max+1]
                heatmap_obj_max = torch.max(heatmap_obj).cpu()

                # object is recognized if maximum pixel value is higher than th
                gt_obj_recogn = gt_obj_max > 0.15
                hm_obj_recogn = heatmap_obj_max > args.threshhold

                hm_max_values.append(heatmap_obj_max)

                if gt_obj_recogn:
                    gt.append(1)
                else:
                    gt.append(0)

                if (hm_obj_recogn and gt_obj_recogn):
                    tp += 1
                elif (hm_obj_recogn and not gt_obj_recogn):
                    fp += 1
                elif (not hm_obj_recogn and gt_obj_recogn):
                    fn += 1

    return tp, fp, fn, all_count


def test(test_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    kld_losses = AverageMeter()

    model.eval()

    tp_total = 0
    fp_total = 0
    fn_total = 0
    all_count_total = 0

    hm_max_values = []
    gt = []

    heightfactor = 576 // 16
    widthfactor = 1024 // 16

    smoothing = GaussianSmoothing(1, 5, 1).cuda(args.gpu)

    with torch.no_grad():
        end = time.time()
        for i, (input, target, gaze_gt, img_names) in enumerate(test_loader):
            # ... (same as before)
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)
                gaze_gt = gaze_gt.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)

            loss = criterion(output, target)

            output = torch.sigmoid(output)

            heatmap = grid_to_heatmap(output,[heightfactor,widthfactor],[16,16],args)
            heatmap = F.interpolate(heatmap, size=[36, 64], mode='bilinear', align_corners=False)
            heatmap = smoothing(heatmap)
            heatmap = F.pad(heatmap, (2, 2, 2, 2), mode='constant')
            heatmap = heatmap.view(heatmap.size(0),-1)
            heatmap = F.softmax(heatmap,dim=1)

            # normalize
            heatmap -= heatmap.min(1, keepdim=True)[0]
            heatmap /= heatmap.max(1, keepdim=True)[0]

            heatmap = heatmap.view(-1,1,36,64)

            for j in range(heatmap.size(0)):
                img_name = img_names[j]
                heatmap_img = heatmap[j]  # predicted gaze map
                gt_img = gaze_gt[j]  # original gaze map

                filename = os.path.join(args.yolo5bb, img_name + ".txt")
                tp, fp, fn, all_count = object_metrics(filename, heatmap_img.cpu(), gt_img.cpu(), gt, hm_max_values, args)

                tp_total += tp
                fp_total += fp
                fn_total += fn
                all_count_total += all_count

                visualize_heatmaps(heatmap_img.cpu(), args.visualizations, img_name)

            # ... (same as before)
            kld = kl(heatmap, gaze_gt)

            losses.update(loss.item(), input.size(0))
            kld_losses.update(kld, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'KL {kl.val:.4f} ({kl.avg:.4f})\t'
                      .format(
                       i, len(test_loader), batch_time=batch_time, loss=losses, kl=kld_losses))
        # ... (same as before)
        print('Test: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'KL {kl.val:.4f} ({kl.avg:.4f})\t'
              .format(
               i, len(test_loader), batch_time=batch_time, loss=losses, kl=kld_losses))
        
        precision = tp_total / (tp_total + fp_total)
        recall = tp_total / (tp_total + fn_total)
        tn_total = all_count_total - tp_total - fp_total - fn_total
        acc = (tp_total+tn_total)/all_count
        f1 = 2*precision*recall/(precision+recall)
        print('Object-level results:')
        print('tp:', tp_total, 'fp:', fp_total, 'tn:', tn_total, 'fn:', fn_total, 'sum:', all_count_total)
        print('prec:', precision, 'recall:', recall, 'f1', f1, 'acc', acc)
        print('AUC:', roc_auc_score(gt, hm_max_values))


def compute_absolute_bounding_box(x_center_rel, y_center_rel, width_rel, height_rel, img_width=64, img_height=36):
    width_abs = width_rel * img_width
    height_abs = height_rel * img_height
    x_center_abs = x_center_rel * img_width
    y_center_abs = y_center_rel * img_height

    x_min, x_max, y_min, y_max = calculate_bounding_box(x_center_abs, y_center_abs, width_abs, height_abs)

    return [max(coord, 0) for coord in [x_min, x_max, y_min, y_max]]


def calculate_bounding_box(x_center_abs, y_center_abs, width_abs, height_abs):
    x_min = int(math.floor(x_center_abs - 0.5 * width_abs))
    x_max = int(math.floor(x_center_abs + 0.5 * width_abs))
    y_min = int(math.floor(y_center_abs - 0.5 * height_abs))
    y_max = int(math.floor(y_center_abs + 0.5 * height_abs))
    return x_min, x_max, y_min, y_max


def grid_to_heatmap(grid, size, num_grid, args):
    heatmap_height, heatmap_width = size[0] * num_grid[0], size[1] * num_grid[1]
    new_heatmap = torch.zeros(grid.size(0), heatmap_height, heatmap_width)

    for i, item in enumerate(grid):
        idx = torch.nonzero(item)
        if idx.nelement() == 0:
            print('Empty')
            continue
        expand_grid(new_heatmap, i, idx, item, num_grid, size)

    output = new_heatmap.unsqueeze(1).cuda(args.gpu)
    return output


def expand_grid(new_heatmap, i, idx, item, num_grid, size):
    for x in idx:
        row_start = x // num_grid[1] * size[0]
        row_end = (x // num_grid[1] + 1) * size[0]
        col_start = x % num_grid[1] * size[1]
        col_end = (x % num_grid[1] + 1) * size[1]
        new_heatmap[i, row_start:row_end, col_start:col_end] = item[x]




def kl(s_map_all, gt_all):
    dims = len(s_map_all.size())
    bs = s_map_all.size()[0]
    eps = torch.tensor(1e-07)
    kl = 0

    if dims > 3:
        kl = kl_divergence(s_map_all, gt_all, bs, eps)

    return kl


def kl_divergence(s_map_all, gt_all, bs, eps):
    kl = 0
    for i in range(bs):
        s_map = s_map_all[i, :, :, :].squeeze()
        gt = gt_all[i, :, :, :].squeeze()
        s_map = s_map / (torch.sum(s_map) * 1.0 + eps)
        gt = gt / (torch.sum(gt) * 1.0 + eps)
        gt = gt.to('cpu')
        s_map = s_map.to('cpu')
        kl += torch.sum(gt * torch.log(eps + gt / (s_map + eps)))
    return kl / bs

def normalize_data(data):
        min_val = torch.min(data)
        max_val = torch.max(data)

        # Normalize data
        normalized_data = (data - min_val) / (max_val - min_val)
        
        return normalized_data

def visualize_heatmaps(heatmap, path, nr):
    heatmap = torchvision.transforms.functional.to_pil_image(heatmap)

    heatmap.save(os.path.join(path, f'{nr}_pred.png'))

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class GaussianSmoothing(nn.Module):
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        kernel_size, sigma = self.prepare_kernel_size_and_sigma(kernel_size, sigma, dim)
        kernel = self.create_gaussian_kernel(kernel_size, sigma)
        self.prepare_and_register_weight(channels, kernel)
        self.groups = channels
        self.conv = self.select_conv_function(dim)

    @staticmethod
    def prepare_kernel_size_and_sigma(kernel_size, sigma, dim):
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim
        return kernel_size, sigma

    @staticmethod
    def create_gaussian_kernel(kernel_size, sigma):
        kernel = 1
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / (2 * std)) ** 2)
        kernel = kernel / torch.sum(kernel)
        return kernel

    def prepare_and_register_weight(self, channels, kernel):
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
        self.register_buffer('weight', kernel)

    @staticmethod
    def select_conv_function(dim):
        if dim == 1:
            return F.conv1d
        elif dim == 2:
            return F.conv2d
        elif dim == 3:
            return F.conv3d
        else:
            raise RuntimeError(f'Only 1, 2 and 3 dimensions are supported. Received {dim}.')

    def forward(self, input):
        return self.conv(input, weight=self.weight, groups=self.groups)


if __name__ == '__main__':
    main()
