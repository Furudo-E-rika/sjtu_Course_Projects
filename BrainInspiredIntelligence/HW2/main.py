import torch
from torch.autograd import Variable
import memtorch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from memtorch.utils import LoadMNIST
import numpy as np
import argparse
from model.resnet import ResNet
from model.cnn1 import CNN1
from model.cnn2 import CNN2
from model.cnn3 import CNN3
from model.cnn_robust import CNN_ROBUST
import copy
from memtorch.mn.Module import patch_model
from memtorch.map.Input import naive_scale
from memtorch.map.Parameter import naive_map
from memtorch.bh.nonideality.NonIdeality import apply_nonidealities

# model testing
def test(model, test_loader, device):
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):        
        output = model(data.to(device))
        pred = output.data.max(1)[1]
        correct += pred.eq(target.to(device).data.view_as(pred)).cpu().sum()

    return 100. * float(correct) / float(len(test_loader.dataset))

def add_noise_to_weights(std, model, device, mean=0):
    """
    with torch.no_grad():
        if hasattr(m, 'weight'):
            m.weight.add_(torch.randn(m.weight.size()) * 0.1)
    """
    gassian_kernel = torch.distributions.Normal(mean, std)
    with torch.no_grad():
        for param in model.parameters():
            param.mul_(torch.exp(gassian_kernel.sample(param.size()).to(device)))
            





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='memristor deep neural network')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--learning_rate', type=float, default=1e-1)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--step_lr', type=str, default=5, help='learning rate decays every step_lr epochs')
    parser.add_argument('--decay_rate', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--model', type=str, default='cnn_robust', help='type of model we use in the classification task')
    parser.add_argument('--add_noise', type=bool, default=True)
    args = parser.parse_args()

    if args.model == 'cnn1': # one conv layer
        model = CNN1(in_channel=1).to(args.device)
    elif args.model == 'cnn2': # two conv layers
        model = CNN2(in_channel=1).to(args.device)
    elif args.model == 'cnn3': # three conv layers
        model = CNN3(in_channel=1).to(args.device)
    elif args.model == 'resnet':
        model = ResNet(input_channel=1, n_classes=10).to(args.device)
    elif args.model == 'cnn_robust':
        model = CNN_ROBUST(in_channel=1).to(args.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    learning_rate = args.learning_rate
    best_accuracy = 0

    ## load mnist dataset
    train_loader, validation_loader, test_loader = LoadMNIST(batch_size=args.batch_size, validation=False)
    for epoch in range(0, args.epochs):
        add_noise_to_weights(std=1, model=model, mean=0, device=args.device)
        print('Epoch: [%d]\t\t' % (epoch + 1), end='')
        if epoch % args.step_lr == 0:
            learning_rate = learning_rate * args.decay_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data.to(args.device))
            loss = criterion(output, target.to(args.device))
            loss.backward()
            optimizer.step()

        accuracy = test(model, test_loader, device=args.device)
        print('%2.2f%%' % accuracy)
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), './trained_models/trained_{}_model.pt'.format(args.model))
            best_accuracy = accuracy

    ## define reference memristor        
    reference_memristor = memtorch.bh.memristor.VTEAM
    reference_memristor_params = {'time_series_resolution': 1e-10}
    memristor = reference_memristor(**reference_memristor_params)
    #emristor.plot_hysteresis_loop()
    #memristor.plot_bipolar_switching_behaviour()

    ## transform DNN to MDNN

    model.load_state_dict(torch.load('./trained_models/trained_{}_model.pt'.format(args.model)), strict=False)
    patched_model = patch_model(copy.deepcopy(model),
                          memristor_model=reference_memristor,
                          memristor_model_params=reference_memristor_params,
                          module_parameters_to_patch=[torch.nn.Conv2d],
                          mapping_routine=naive_map,
                          transistor=True,
                          programming_routine=None,
                          tile_shape=(128, 128),
                          max_input_voltage=0.3,
                          scaling_routine=naive_scale,
                          ADC_resolution=8,
                          ADC_overflow_rate=0.,
                          quant_method='linear')

    patched_model_ = apply_nonidealities(copy.deepcopy(patched_model),
                                     non_idealities=[memtorch.bh.nonideality.NonIdeality.DeviceFaults],
                                     lrs_proportion=0.25,
                                     hrs_proportion=0.10,
                                     electroform_proportion=0)
    patched_model.tune_()
    add_noise_to_weights(std=0.5, model=patched_model, mean=0, device=args.device)

    

    print(test(patched_model, test_loader, device=args.device))

    
