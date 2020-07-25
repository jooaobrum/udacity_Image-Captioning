import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.embed_size = embed_size
        
        # Transform the caption into embedding
        self.embed_caption = nn.Embedding(vocab_size, embed_size)
        # Intialize LSTM cells
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first = True)
        # Linear layer
        self.linear_layer = nn.Linear(hidden_size, vocab_size)

    
    def forward(self, features, captions):
        # Transform the captions to embeddings
        embeded=self.embed_caption(captions[:,:-1])
        
        # Transform features to torch
        input_embed=torch.cat((features.unsqueeze(dim=1),embeded),dim=1)
        
        # output lstm
        lstm_output,_ =self.lstm(input_embed)
        
        # lstm passing through linear layer
        output=self.linear_layer(lstm_output)
        # now return the output
        return output
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        generated_caption = []
        
        for i in range(max_len):
            out_lstm, states = self.lstm(inputs, states)
            
            outputs = self.linear_layer(out_lstm.squeeze(1))
            _, output = outputs.max(dim=1)                   
           
            # Append the list
            generated_caption.append(output.item())
            
            # for next iteration
            inputs = self.embed_caption(output)             
            inputs = inputs.unsqueeze(1)   
        return generated_caption