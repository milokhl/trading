import sys, os
import numpy as np

import torch.nn.functional as F
from torch import nn, FloatTensor
from torch.autograd import Variable
from torch.optim import Adam, SGD

from torchnet.engine import Engine
from torchnet.logger import VisdomPlotLogger, VisdomLogger
from tqdm import tqdm
import torchnet as tnt

sys.path.append('../')
from event_dataset import *
from event_embedding_model import EventEmbedder

LOAD_EPOCH = None # './epochs/epoch_7.pt'
LOAD_EPOCH_NUM = 0
INPUT_TENSOR_DIR = '../input_tensors/'

BATCH_SIZE = 512
NUM_EPOCHS = 50

if __name__ == "__main__":

    # Create the model.
    model = EventEmbedder(300, 50, 32)

    # Load from checkpoint here is needed.
    if LOAD_EPOCH != None:
        print('Loading model from epoch:', LOAD_EPOCH)
        model.load_state_dict(torch.load(LOAD_EPOCH))
    model.cuda() # CUDA that shit.
    print("Model Parameters:", sum(param.numel() for param in model.parameters()))

    # Create the optimizer.
    optimizer = Adam(model.parameters())

    # Create the torchnet engine and metrics.
    engine = Engine()
    meter_loss = tnt.meter.AverageValueMeter()

    # Create a bunch of loggers that can be viewed in the browser.
    train_loss_logger = VisdomPlotLogger('line', opts={'title': 'Train Loss'})
    test_loss_logger = VisdomPlotLogger('line', opts={'title': 'Test Loss'})

    # Define the loss function.
    # http://pytorch.org/docs/0.1.12/_modules/torch/nn/modules/loss.html
    lossFn = nn.MarginRankingLoss(size_average=True)


    def get_iterator(mode):
        """
        Returns an iterable DataLoader object.
        @param mode (bool) True for training mode, False for testing mode.
        """
        if mode:
            pass
        else:
            pass

        dataset = EventDataset(tensor_dir=INPUT_TENSOR_DIR, memory_limit=2e9)
        return torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)


    def processor(sample):
        """
        A function that takes in a sample of data and returns the network loss and outputs.
        Called inside the training and testing loops of the engine.
        """
        O1 = Variable(sample[:,0,:]).cuda()
        P = Variable(sample[:,1,:]).cuda()
        O2 = Variable(sample[:,2,:]).cuda()

        O1_c = Variable(sample[:,3,:]).cuda()
        P_c = Variable(sample[:,4,:]).cuda()
        O2_c = Variable(sample[:,5,:]).cuda()

        real_score, real_embed = model(O1, P, O2)
        corr_score, corr_embed = model(O1_c, P_c, O2_c)

        # Argument of 1 means that real_score should be higher than corr_score.
        loss = lossFn(real_score, corr_score, Variable(torch.ones(1, real_score.size()[0])).cuda())
        return (loss, (real_score, corr_score))

    # Called before the start of a training or validation epoch.
    def reset_meters():
        meter_loss.reset()


    # Called every time the training loop requests a new batch.
    def on_sample(state):
        pass


    # Called every time a batch is feed forward through the model.
    def on_forward(state, train_log_freq=10):
        meter_loss.add(state['loss'].data[0])

        if state['t'] % train_log_freq == 0:
            train_loss_logger.log(state['t'], meter_loss.value()[0])

    # Called at the start of each new epoch.
    def on_start_epoch(state):
        reset_meters()
        state['iterator'] = tqdm(state['iterator'])


    # Called at the end of every epoch.
    def on_end_epoch(state):
        reset_meters()

        # Validate the model after every training epoch.
        engine.test(processor, get_iterator(False))
        test_loss_logger.log(state['epoch'], meter_loss.value()[0])

        print('[Epoch %d] Testing Loss: %.4f' % (state['epoch'], meter_loss.value()[0]))

        print('Saving model checkpoint: epochs/epoch_%d.pt' % state['epoch'])
        torch.save(model.state_dict(), 'epochs/epoch_%d.pt' % state['epoch'])

    # Called when the engine first starts up the training and testing loops.
    def on_start(state):
        # Want to only set the epoch when entering the training loop.
        # Otherwise, epoch saves will get overwritten.
        if LOAD_EPOCH != None and state['train'] == True:
            print('Setting the state epoch to:', LOAD_EPOCH_NUM)
            state['epoch'] = LOAD_EPOCH_NUM

    # Set up hooks for the engine.
    engine.hooks['on_start'] = on_start
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch

    # Network feedforward function, DataLoader object, maxepoch, optimizer
    engine.train(processor, get_iterator(True), maxepoch=NUM_EPOCHS, optimizer=optimizer)
