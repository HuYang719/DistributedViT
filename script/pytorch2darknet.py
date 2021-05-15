import torch
import torchvision
from cfg import save_conv, save_conv_bn, save_fc
import os

def save_bottlenet_weights(model, fp):
    save_conv_bn(fp, model.conv1, model.bn1)
    save_conv_bn(fp, model.conv2, model.bn2)
    save_conv_bn(fp, model.conv3, model.bn3)
    if model.downsample:
        save_conv_bn(fp, model.downsample[0], model.downsample[1])

def save_resnet_weights(model, filename):
    fp = open(filename, 'wb')
    header = torch.IntTensor([0,0,0,0])
    header.numpy().tofile(fp)
    save_conv_bn(fp, model.conv1, model.bn1)
    for i in range(len(model.layer1._modules)):
        save_bottlenet_weights(model.layer1[i], fp)
    for i in range(len(model.layer2._modules)):
        save_bottlenet_weights(model.layer2[i], fp)
    for i in range(len(model.layer3._modules)):
        save_bottlenet_weights(model.layer3[i], fp)
    for i in range(len(model.layer4._modules)):
        save_bottlenet_weights(model.layer4[i], fp)
    save_fc(fp, model.fc)
    fp.close()

def save_vgg16_weights(model, filename):
    fp = open(filename, 'wb')
    header = torch.IntTensor([0,0,0,0])
    header.numpy().tofile(fp)
    for layer in model.features:
        if type(layer) == torch.nn.Conv2d:
            # print(layer)
            save_conv(fp, layer)
    for layer in model.classifier:
        if type(layer) == torch.nn.Linear:
            # print(layer)
            save_fc(fp, layer)

def save_classtoken(fp, model):
    model.class_token.data.numpy().tofile(fp)


def save_posemb(fp, model):
    model.pos_embedding.data.numpy().tofile(fp)
   
    # print(model.pos_embedding.data.numpy())

def save_layernorm(fp, norm, index):
    norm.bias.data.numpy().tofile(fp)
    norm.weight.data.numpy().tofile(fp)
    if(index == 0):
        print("layer norm shape is {}, bias is {}".format( norm.bias.data.numpy().shape, norm.bias.data.numpy()))
        print("layer norm shape is {}, weight is {}".format( norm.weight.data.numpy().shape, norm.weight.data.numpy()))

def save_multihead(fp, attn, flag):
    # heads = [attn.proj_q, attn.proj_k, attn.proj_v]
    # for head in heads:
    #     save_fc(fp, head, 1)
    save_fc(fp, attn.proj_q, 0)
    save_fc(fp, attn.proj_k, 0)
    save_fc(fp, attn.proj_v, flag)

def save_feedfwd(fp, pwff, id):
    # layers = [pwff.fc1, pwff.fc2]
    save_fc(fp, pwff.fc1, id)
    save_fc(fp, pwff.fc2, 0)

def save_transformer(module, fp):
    #parse wrap
    modules = module.blocks.named_children()
    for name,module in modules:
        #module includes:
        #   (norm1) LayerNorm 1
        #   (attn)  MultiHead: linear q,k,v + dropout
        #   (proj)  Linear
        #   (drop)  Dropout + shorcut
        #   (norm2) LayerNorm 2
        #   (pwff)  FeedForward: linear 1,2
        #   (drop)  Dropout
        if(name == '0'):
            save_layernorm(fp, module.norm1, 1)
        else:
            save_layernorm(fp, module.norm1, 1)
        if name == '0':
            save_multihead(fp, module.attn, 0)
        else:
            save_multihead(fp, module.attn, 0)
        if name == '0':
            save_fc(fp, module.proj, 0)
        else:
            save_fc(fp, module.proj)
        save_layernorm(fp, module.norm2, 1)
        if name == '0':
            save_feedfwd(fp, module.pwff, 1)
        else:
            save_feedfwd(fp, module.pwff, 0)
            

def save_vit_weights(model, filename):
    with open(filename, 'wb') as fp:
        header = torch.IntTensor([0,0,0,0])
        header.numpy().tofile(fp)
        #save patch embedding
        save_conv(fp, model.patch_embedding)
        #save positional embedding
        save_posemb(fp, model.positional_embedding)
        #class token 
        save_classtoken(fp, model)
        #save transformer
        save_transformer(model.transformer, fp)
        save_layernorm(fp, model.norm, 1)
        save_fc(fp, model.fc)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', '-a', metavar='ARCH', default='vit',
                    #choices=model_names,
                    help='model architecture: ' +
                        ' (default: vit)')
    args = parser.parse_args()


    model_name = args.arch
    if model_name == 'resnet50':
        resnet50 = torchvision.models.resnet50(pretrained=True)
        print('convert pytorch resnet50 to darkent, save resnet50.weights')
        save_resnet_weights(resnet50, 'resnet50.weights')
    elif model_name == 'vgg16':
        vgg16 = torchvision.models.vgg16(pretrained=True)
        print('convert pytorch vgg16 to darkneg, save vgg16-pytorch2darknet.weights')
        save_vgg16_weights(vgg16, 'vgg16-pytorch2darknet.weights')
    elif model_name == 'vit':
        from pytorch_pretrained_vit import ViT
        vit = ViT('B_16_imagenet1k', pretrained=True)
        file_name = 'vit2.weights'
        if os.path.exists(file_name):
            os.remove(file_name)
        print('convert pytorch ViT to darknet, save ' + file_name)
        save_vit_weights(vit, file_name)
