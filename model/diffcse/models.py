import torch

from torch import nn
from einops import repeat
from transformers import (
        AutoModel, 
        AutoConfig, 
        AutoModelForMaskedLM,
        BertForPreTraining
)
from model.diffcse.modeling_bert import BertModel
from model.diffcse.modeling_roberta import RobertaModel


class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x


class ProjectionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_dim = config.hidden_size
        self.hidden_dim = config.hidden_size * 2
        self.out_dim = config.hidden_size
        affine=False
        list_layers = [nn.Linear(self.in_dim, self.hidden_dim, bias=False),
                       nn.BatchNorm1d(self.hidden_dim),
                       nn.ReLU(inplace=True)]
        list_layers += [nn.Linear(self.hidden_dim, self.out_dim, bias=False),
                        nn.BatchNorm1d(self.out_dim, affine=affine)]
        self.net = nn.Sequential(*list_layers)

    def forward(self, x):
        return self.net(x)
        

class DiffCSE(nn.Module):
    def __init__(self, args, tokenizer):
        super(DiffCSE, self).__init__()

        self.args = args
        self.config = AutoConfig.from_pretrained(self.args.model)
       
        self.init_token = tokenizer.cls_token
        self.init_token_idx = tokenizer.convert_tokens_to_ids(self.init_token)
        
        if 'roberta' in self.args.model:
            self.encoder = RobertaModel.from_pretrained(self.args.model)
            self.discriminator = RobertaModel.from_pretrained(self.args.model)
        elif 'bert' in self.args.model:
            self.encoder = BertModel.from_pretrained(self.args.model)
            self.discriminator = BertModel.from_pretrained(self.args.model)
        else:
            raise NotImplementedError

        self.lm_head = nn.Linear(self.config.hidden_size, 2)
        cl_init(self, self.args, self.config)

    def forward(self, config, inputs, mode):

        if mode == 'train':
            _, z1_pooler_output = self.encoder(input_ids=inputs['anchor']['source'],
                                               token_type_ids=inputs['anchor']['token_type_ids'],
                                               attention_mask=inputs['anchor']['attention_mask'],
                                               return_dict=False)
            
            _, z2_pooler_output = self.encoder(input_ids=inputs['anchor']['source'],
                                               token_type_ids=inputs['anchor']['token_type_ids'],
                                               attention_mask=inputs['anchor']['attention_mask'],
                                               return_dict=False)
            
            g_pred = self.generator_outputs(inputs)
            
            z1_replaced = (g_pred[0] != inputs['anchor']['source']) * inputs['anchor']['attention_mask']
            z2_replaced = (g_pred[1] != inputs['anchor']['source']) * inputs['anchor']['attention_mask']
            
            z1_e_inputs = g_pred[0] * inputs['anchor']['attention_mask']
            z2_e_inputs = g_pred[1] * inputs['anchor']['attention_mask']

            mlm_outputs = self.discriminator_outputs(inputs,
                                                     z1_e_inputs,
                                                     z2_e_inputs,
                                                     z1_pooler_output,
                                                     z2_pooler_output)
            
            z1_e_labels = z1_replaced.view(-1, z1_replaced.size(-1))
            z2_e_labels = z2_replaced.view(-1, z2_replaced.size(-1))

            z1_electra_acc = float(
                    ((mlm_outputs[0].argmax(-1) == z1_e_labels) * inputs['anchor']['attention_mask']).sum()
                    /inputs['anchor']['attention_mask'].sum())
            z2_electra_acc = float(
                    ((mlm_outputs[1].argmax(-1) == z2_e_labels) * inputs['anchor']['attention_mask']).sum()
                    /inputs['anchor']['attention_mask'].sum())
            
            electra_acc = (z1_electra_acc + z2_electra_acc) / 2
            
            diff_outputs = [mlm_outputs, z1_e_labels, z2_e_labels, torch.FloatTensor([electra_acc]).to(self.args.device)]
            
            return z1_pooler_output, z2_pooler_output, diff_outputs

        else:
            sentence_1_pooler, _ = self.encoder(input_ids=inputs['sentence_1']['source'],
                                                token_type_ids=inputs['sentence_1']['token_type_ids'],
                                                attention_mask=inputs['sentence_1']['attention_mask'],
                                                return_dict=False)

            sentence_2_pooler, _ = self.encoder(input_ids=inputs['sentence_2']['source'],
                                                token_type_ids=inputs['sentence_2']['token_type_ids'],
                                                attention_mask=inputs['sentence_2']['attention_mask'],
                                                return_dict=False)

            return sentence_1_pooler[:, 0], sentence_2_pooler[:, 0]

    
    def generator_outputs(self, inputs):
        with torch.no_grad():
            z1_g_pred = self.generator(input_ids=inputs['anchor']['z1_masked_input'],
                                       token_type_ids=inputs['anchor']['token_type_ids'],
                                       attention_mask=inputs['anchor']['attention_mask'])
            
            z2_g_pred = self.generator(input_ids=inputs['anchor']['z2_masked_input'],
                                       token_type_ids=inputs['anchor']['token_type_ids'],
                                       attention_mask=inputs['anchor']['attention_mask'])
               
        z1_g_pred = z1_g_pred[0].argmax(-1)
        z2_g_pred = z2_g_pred[0].argmax(-1)

        z1_g_pred[:, 0] = self.init_token_idx
        z2_g_pred[:, 0] = self.init_token_idx
        
        return (z1_g_pred, z2_g_pred)
            
    def discriminator_outputs(self, 
                              inputs=None, 
                              z1_e_inputs=None, 
                              z2_e_inputs=None,
                              z1_pooler_output=None,
                              z2_pooler_output=None,
                              ):
        
        z1_mlm_outputs = self.discriminator(z1_e_inputs,
                                            token_type_ids=inputs['anchor']['token_type_ids'],
                                            attention_mask=inputs['anchor']['attention_mask'],
                                            cls_input=z1_pooler_output.view((-1, z1_pooler_output.size(-1)))
        
        )
        z2_mlm_outputs = self.discriminator(z2_e_inputs,
                                            token_type_ids=inputs['anchor']['token_type_ids'],
                                            attention_mask=inputs['anchor']['attention_mask'],
                                            cls_input=z2_pooler_output.view((-1, z2_pooler_output.size(-1)))

        )
            
        z1_pred_scores = self.lm_head(z1_mlm_outputs.last_hidden_state)
        z2_pred_scores = self.lm_head(z2_mlm_outputs.last_hidden_state)

        return (z1_pred_scores, z2_pred_scores)

    def encode(self, inputs, device):

        embeddings, _ = self.bert(input_ids=inputs['source'].to(device),
                                  token_type_ids=inputs['token_type_ids'].to(device),
                                  attention_mask=inputs['attention_mask'].to(device),
                                  return_dict=False)
        
        return embeddings[:, 0]

def cl_init(cls, args, config):
    cls.mlp = ProjectionMLP(config)
    cls.generator = AutoModelForMaskedLM.from_pretrained(args.generator_name)
