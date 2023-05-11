from __future__ import print_function

import argparse
import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from torchfusion.gan.applications import DCGANDiscriminator

from data_loader import MarioDataset
from models.custom import Generator


# Run with "python main.py"
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nz", type=int, default=32, help="size of the latent z vector"
    )
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument("--batchSize", type=int,
                        default=32, help="input batch size")
    parser.add_argument(
        "--niter", type=int, default=10000, help="number of epochs to train for"
    )
    parser.add_argument(
        "--lrD",
        type=float,
        default=0.00005,
        help="learning rate for Critic, default=0.00005",
    )
    parser.add_argument(
        "--lrG",
        type=float,
        default=0.00005,
        help="learning rate for Generator, default=0.00005",
    )
    parser.add_argument(
        "--beta1", type=float, default=0.5, help="beta1 for adam. default=0.5"
    )
    parser.add_argument("--cuda", action="store_true", help="enables cuda")
    parser.add_argument("--ngpu", type=int, default=1,
                        help="number of GPUs to use")
    parser.add_argument(
        "--netG", default="", help="path to netG (to continue training)"
    )
    parser.add_argument(
        "--netD", default="", help="path to netD (to continue training)"
    )
    parser.add_argument("--clamp_lower", type=float, default=-0.01)
    parser.add_argument("--clamp_upper", type=float, default=0.01)
    parser.add_argument(
        "--Diters", type=int, default=5, help="number of D iters per each G iter"
    )

    parser.add_argument(
        "--n_extra_layers",
        type=int,
        default=0,
        help="Number of extra layers on gen and disc",
    )
    parser.add_argument(
        "--experiment", default=None, help="Where to store samples and models"
    )
    parser.add_argument(
        "--adam", action="store_true", help="Whether to use adam (default is rmsprop)"
    )
    parser.add_argument("--problem", type=int,
                        default=0, help="Level examples")
    opt = parser.parse_args()
    return opt


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def tiles2image(tiles, z_dims):
    '''
    Plotting the image from numeric encodings at this stage is just using the 
    color map to visualize different color. 
    '''
    return plt.get_cmap("rainbow")(tiles / float(z_dims))


def combine_images(generated_images):
    '''
    the generated_images is the output of tiles2image
    '''
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = generated_images.shape[1:]
    image = np.zeros(
        (height * shape[0], width * shape[1], shape[2]), dtype=generated_images.dtype
    )
    for index, img in enumerate(generated_images):
        i = int(index / width)
        j = index % width
        image[
            i * shape[0]: (i + 1) * shape[0], j * shape[1]: (j + 1) * shape[1]
        ] = img
    return image


def train(
    netG, netD, org_data, opt, nz, z_dims, map_size, num_batches, conditional_channels
):
    '''
    netG: generator
    netD: discrimminator
    org_data: from MarioDataset(), size = (2488,13,32,32), where (2488,13,9:-9, 2:-2) is the onehot encoding and others are zeros
    opt: whether using cuda
    nz: 14 * 32 * 32,
    z_dim: number of mapped symbols (tile types)
    map_size: 32
    batchsize: default 32

    The condition of the DCGAN is the previous frame
    '''
    input = torch.FloatTensor(opt.batchSize, z_dims, map_size, map_size)
    noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
    # fill the tensor with elements samples from the normal distribution
    fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
    one = torch.FloatTensor([1])
    mone = one * -1

    if opt.cuda:
        netD.cuda()
        netG.cuda()
        input = input.cuda()
        one, mone = one.cuda(), mone.cuda()
        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
        org_data.cuda()

    # setup optimizer
    if opt.adam:
        optimizerD = optim.Adam(
            netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999))
        optimizerG = optim.Adam(
            netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999))
        print("Using ADAM")
    else:
        optimizerD = optim.RMSprop(netD.parameters(), lr=opt.lrD)
        optimizerG = optim.RMSprop(netG.parameters(), lr=opt.lrG)

    gen_iterations = 0
    for epoch in range(opt.niter):

        #! data_iter = iter(dataloader)
        # random permuation of intergers: generate random interger without repeating
        data_idx = torch.randperm(len(org_data))

        i = 0
        while i < num_batches:  # len(dataloader):
            ############################
            # (1) Update D network
            ###########################
            for p in netD.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update

            # train the discriminator Diters times
            if gen_iterations < 25 or gen_iterations % 500 == 0:
                Diters = 100  # what is Diter for?
            else:
                Diters = opt.Diters
            j = 0
            while j < Diters and i < num_batches:  # len(dataloader):
                j += 1

                # clamp parameters to a cube
                for p in netD.parameters():
                    # the value < lb will be replaced by lb and > ub by ub
                    # why do they add this?
                    p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

                batch_data = org_data[
                    data_idx[i * opt.batchSize: (i + 1) * opt.batchSize]
                ]  # generate batch data from original dataset

                i += 1
                context_frame, out_frame = (batch_data[0], batch_data[1])

                if False:
                    # im = data.cpu().numpy()
                    print(batch_data.shape)
                    real_cpu = combine_images(
                        tiles2image(np.argmax(batch_data, axis=1),
                                    z_dims=z_dims)
                    )
                    print(real_cpu)
                    plt.imsave(
                        "{0}/real_samples.png".format(opt.experiment), real_cpu)
                    exit()

                netD.zero_grad()
                # batch_size = num_samples #real_cpu.size(0)

                if opt.cuda:
                    context_frame.cuda(), out_frame.cuda()
                joined_frame = torch.cat((context_frame, out_frame), dim=3)
                # the size of every joined_frame (for a single level) is (13,32,32)
                assert joined_frame[0].shape == (13, 32, 32)
                input.resize_as_(joined_frame).copy_(
                    joined_frame)
                inputv = Variable(input)  # make input as variables wrt loss

                # forward the batch sample through netD and get error: the corresponding label shoule be "real"
                errD_real = netD(inputv).mean(0).view(1)
                errD_real.backward(one)

                # train with fake
                noise.resize_(opt.batchSize, 1, 14, 14).normal_(0, 1)

                ref_idx = torch.randperm(len(org_data))[: opt.batchSize]
                # only take take out the first half of the frame
                ref_frames = org_data[ref_idx].prev_frame
                gen_input = torch.cat(
                    (noise, ref_frames[:,
                     conditional_channels, 9:-9, 2:]), dim=1  # conditional channel = [0,1,6,7]
                )
                # gen_input.size() = [opt.batchSize,5,14,14]
                # gen_input = ref_frames[:, :, 9:-9, 2:]

                # totally freeze netG
                noisev = Variable(gen_input, volatile=True) # what is volatile=True?
                # generated latent variables from netG, **size = (batchSize,?,32,32)
                fake = Variable(netG(noisev).data)
                stitched = torch.cat(
                    (ref_frames, fake[:, :, :, 16:]), dim=3
                )  # stitch the first half and the generated fake second have together
                assert stitched[0].shape == (13, 32, 32)
                inputv = stitched
                errD_fake = netD(inputv).mean(0).view(
                    1)  # tell netD this is the fake one.
                # get gradient for the fake data. mone is to tell netD this is the fake model
                errD_fake.backward(mone)
                errD = errD_real - errD_fake  # To track error difference
                optimizerD.step()

            ############################
            # (2) Update G network
            ###########################

            # For each batch
            for p in netD.parameters():
                p.requires_grad = False  # Freeze netD, to avoid computation
            netG.zero_grad()
            # in case our last batch was the tail batch of the dataloader,
            # make sure we feed a full batch of noise
            noise.resize_(opt.batchSize, 1, 14, 14).normal_(0, 1)

            ref_idx = torch.randperm(len(org_data))[: opt.batchSize]
            ref_frames = org_data[ref_idx].prev_frame

            gen_input = torch.cat(
                (noise, ref_frames[:, conditional_channels, 9:-9, 2:]), dim=1
            )
            # gen_input = ref_frames[:, :, 9:-9, 2:]

            g_input = Variable(gen_input)
            fake = netG(g_input)
            # fake[:, :, :, :16] = ref_frames
            stitched = torch.cat((ref_frames, fake[:, :, :, 16:]), dim=3)
            errG = netD(stitched).mean(0).view(1)
            errG.backward(one)
            optimizerG.step()
            gen_iterations += 1
            # 0430/2023
            print(
                "[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f"
                % (
                    epoch,
                    opt.niter,
                    i,
                    num_batches,
                    gen_iterations,
                    errD.data[0],
                    errG.data[0],
                    errD_real.data[0],
                    errD_fake.data[0],

                )
            )

            # np.savetxt("Loss_D.csv", np.asarray(errD.data[0]), delimiter = ",")
            # np.savetxt("Loss_G.csv", np.asarray(errG.data[0]), delimiter = ",")
            # np.savetxt("Loss_D_real.csv", np.asarray(errD_real.data[0]), delimiter = ",")
            # np.savetxt("Loss_D_fake.csv", np.asarray(errD_fake.data[0]), delimiter = ",")

            # What is this part doing?
            if gen_iterations % 10000 == 0:  # was 500
                # iteration through 10000 batches
                with torch.no_grad():
                    fixed_noise.resize_(opt.batchSize, 1, 14, 14)
                    ref_idx = torch.randperm(len(org_data))[: opt.batchSize]
                    ref_frames = org_data[ref_idx].prev_frame
                    gen_input = torch.cat(
                        (fixed_noise,
                         ref_frames[:, conditional_channels, 9:-9, 2:]),
                        dim=1,
                    )
                    # gen_input = ref_frames[:, :, 9:-9, 2:]
                    fake = netG(Variable(gen_input, volatile=True))
                stitched = torch.cat((ref_frames, fake[:, :, :, 16:]), dim=3)
                im = stitched.data.cpu().numpy()
                im = im[:, :, 9:-9, 1:-2]  # why -1:2 not -2:2
                # print('SHAPE fake',type(im), im.shape)
                # print('SUM ',np.sum( im, axis = 1) )

                im = combine_images(tiles2image(
                    np.argmax(im, axis=1), z_dims=13))  # why argmax?
                plt.imsave(
                    "{0}/mario_fake_samples_1_{1}.png".format(
                        opt.experiment, gen_iterations
                    ),
                    im,
                )
                torch.save(
                    netG.state_dict(),
                    "{0}/netG_epoch_condition_1_{1}_{2}_{3}.pth".format(
                        opt.experiment, gen_iterations, opt.problem, opt.nz
                    ),
                )

        # do checkpointing
        # torch.save(netG.state_dict(), '{0}/netG_epoch_{1}.pth'.format(opt.experiment, epoch))
        # torch.save(netD.state_dict(), '{0}/netD_epoch_{1}.pth'.format(opt.experiment, epoch))


def main():
    opt = parse_arguments()

    if opt.experiment is None:
        opt.experiment = "samples"
    os.system("mkdir {0}".format(opt.experiment))

    opt.manualSeed = random.randint(1, 10000)  # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    map_size = 32
    data = MarioDataset()
    # channels on which generator is conditioned on
    conditional_channels = [0,1,3,4,5,6,7,10,11,12]

    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    ndf = int(opt.ndf)

    # n_extra_layers = int(opt.n_extra_layers)

    # netG = dcgan.DCGAN_G(map_size, nz, z_dims, ngf, ngpu, n_extra_layers)
    netG = Generator(
        latent_size=(len(conditional_channels) + 1, 14, 14), out_size=(13, 32, 32)
    )
    # what does the latent size means? the dimension of the inputs
    print(netG)
    # netG.apply(weights_init)
    # if opt.netG != "":  # load checkpoint if needed
    #     netG.load_state_dict(torch.load(opt.netG))

    netD = DCGANDiscriminator(input_size=(13, 32, 32), apply_sigmoid=False)
    print(netD)
    # netD.apply(weights_init)
    # if opt.netD != "":
    # netD.load_state_dict(torch.load(opt.netD))
    num_batches = len(data) / opt.batchSize
    train(
        netG=netG,
        netD=netD,
        org_data=data,
        opt=opt,
        nz=14 * 32 * 32,
        z_dims=14,
        map_size=32,
        num_batches=num_batches,
        conditional_channels=conditional_channels,
    )


if __name__ == "__main__":
    main()
