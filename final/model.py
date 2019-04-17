import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import ipdb

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        vgg16 = models.vgg16(pretrained=True)

        num_features = vgg16.classifier[-1].in_features
        classifier = list(vgg16.classifier.children())[:-1]
        vgg16.classifier = nn.Sequential(*classifier)
        self.vgg16 = vgg16
        self.linear = nn.Linear(num_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, img):
        with torch.no_grad():
            features = self.vgg16(img)
        features = nn.Dropout(0.5)(features)
        features = self.bn(self.linear(features))
        return features

class EncoderRNN(nn.Module):
    def __init__(self, num_objects, embed_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.embedding = nn.Embedding(num_objects, embed_size)
        self.LSTM = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, embed_size)

    def forward(self, input, h0, c0):
        embed = self.embedding(input)
        embed = nn.Dropout(0.5)(embed)
        output, (h0, c0) = self.LSTM(embed, (h0, c0))
        output = nn.Dropout(0.5)(output)
        output = self.linear(output)
        output = nn.Dropout(0.5)(output)
        return output, (h0, c0), self.embedding.weight

class Model(nn.Module):
    def __init__(self, num_objects, embed_size):
        super(Model, self).__init__()
        self.linear = nn.Linear(embed_size, embed_size)
        self.out_layer = nn.Linear(embed_size, num_objects)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, image_features, hashtag_features, Ul):
        output = self.linear(image_features + hashtag_features.squeeze(1))
        output = nn.Dropout(0.5)(output)
        output = torch.mm(output, Ul.transpose(0,1))
        output = self.softmax(output)
        return output

