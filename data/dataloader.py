import time
import numpy
import torch
import random
import logging

from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer 

logger = logging.getLogger(__name__)


class ModelDataLoader(Dataset):
    def __init__(self, file_path, args, metric, tokenizer, type_):
        self.type = type_
        self.args = args
        self.metric = metric

        """Unsup"""
        self.input_ids = None
        self.token_type_ids = None
        self.attention_mask = None

        self.z1_masked_input = None
        self.z2_masked_input = None

        """STS"""
        self.label = []
        self.sentence_1 = []
        self.sentence_2 = []

        #  -------------------------------------
        self.processes = []
        self.missing_value = 0
        self.bert_tokenizer = tokenizer
        self.file_path = file_path

        """
        [CLS]: 2
        [PAD]: 0
        [UNK]: 1
        """
        
        self.init_token = self.bert_tokenizer.cls_token
        self.pad_token = self.bert_tokenizer.pad_token
        self.unk_token = self.bert_tokenizer.unk_token
        self.mask_token = self.bert_tokenizer.sep_token

        self.init_token_idx = self.bert_tokenizer.convert_tokens_to_ids(self.init_token)
        self.pad_token_idx = self.bert_tokenizer.convert_tokens_to_ids(self.pad_token)
        self.unk_token_idx = self.bert_tokenizer.convert_tokens_to_ids(self.unk_token)
        self.mask_token_idx = self.bert_tokenizer.convert_tokens_to_ids(self.mask_token)
        print(f"cls: {self.init_token_idx}")
        print(f"pad: {self.pad_token_idx}")
        print(f"mask: {self.mask_token_idx}")
        
    def load_data(self, type):
        with open(self.file_path) as file:
            lines = file.readlines()
            
            if type == 'train':
                start_time = time.time()
                
                logger.info("Preprocessing Training Data...")
                self.data2tensor(lines, type)

                end_time = time.time()
                epoch_mins, epoch_secs = self.metric.cal_time(start_time, end_time)
            
                logger.info(f"Complete Preprocessing {epoch_mins}m {epoch_secs}s")
                
            else:
                for _, line in enumerate(tqdm(lines)):
                    self.data2tensor(line, type)
    
        if type == 'train':
            assert len(self.input_ids) == len(self.z1_masked_input) == len(self.z2_masked_input)
        else:
            assert len(self.sentence_1) == len(self.sentence_2) == len(self.label)

    def data2tensor(self, line, type):

        if type == 'train':
            anchor_sen = line
            anchor = self.bert_tokenizer(anchor_sen, 
                                         truncation=True,
                                         return_tensors="pt",
                                         max_length=self.args.max_len,
                                         padding='max_length')
                
            self.input_ids = anchor['input_ids'].numpy()
            self.token_type_ids = anchor['token_type_ids'].numpy()
            self.attention_mask = anchor['attention_mask'].numpy()
            
            logger.info("Masking Training Data...")

            for idx in range(2):
                self.only_mask_tokens(anchor['input_ids'].clone(), anchor['attention_mask'].clone(), idx)
                
            logger.info(f"Missing Value {int(self.missing_value/2)}")

        else:
            
            split_data = line.split('\t')
            sentence_1, sentence_2, label = split_data

            sentence_1 = self.bert_tokenizer(sentence_1, 
                                             truncation=True,
                                             return_tensors="pt",
                                             max_length=self.args.max_len,
                                             padding='max_length')

            sentence_2 = self.bert_tokenizer(sentence_2,
                                             truncation=True,
                                             return_tensors="pt",
                                             max_length=self.args.max_len,
                                             padding='max_length')

            self.sentence_1.append(sentence_1)
            self.sentence_2.append(sentence_2)
            self.label.append(float(label.strip())/5.0)

    def only_mask_tokens(self, inputs, attn_mask, idx):
        cur_inputs = inputs.clone()

        probability_matrix = torch.full(cur_inputs.shape, self.args.masking_ratio)

        special_tokens_mask = [
                self.bert_tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in cur_inputs.tolist()]

        # tensor([ True, False, False, False, False, False, False,  True,  True])
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        
        masking_map = torch.zeros(cur_inputs.shape)
        value_map = torch.ones(cur_inputs.shape).bool()

        for step, source in enumerate(special_tokens_mask):
            
            num_of_false = self.args.max_len - source.sum()
            num_of_mask = max(1, int(num_of_false * self.args.masking_ratio))
            
            if num_of_false != 0:
                random_list = torch.FloatTensor([random.randint(1, num_of_false) for _ in range(num_of_mask)])
                duplicate = random_list.size() != random_list.unique().size()
                while(duplicate):
                    random_list = torch.FloatTensor([random.randint(1, num_of_false) for _ in range(num_of_mask)])
                    duplicate = random_list.size() != random_list.unique().size()
                    
                index = torch.arange(num_of_mask)    
                masking_map[step].index_add_(0, index, random_list)
            else:
                index = torch.arange(num_of_mask)
                masking_map[step].index_add_(0, index, torch.FloatTensor([0]))
                self.missing_value += 1

        masking_map = masking_map.type(torch.int64)
        special_tokens_mask.scatter_(1, masking_map, value_map)
    
        inputs[special_tokens_mask] = self.mask_token_idx
        
        inputs *= attn_mask
        inputs[inputs==0] = self.pad_token_idx
        inputs[:, 0] = self.init_token_idx
        
        if idx == 0: self.z1_masked_input = inputs.numpy()
        else: self.z2_masked_input = inputs.numpy()
        
    def torch_mask_tokens(self, inputs, idx):
        labels = inputs.clone()
    
        # tensor([0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500])
        probability_matrix = torch.full(labels.shape, self.args.masking_ratio)
    
        special_tokens_mask = [
                self.bert_tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]
        
        # tensor([ True, False, False, False, False, False, False,  True,  True])
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

        # tensor([0.0000, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.0000, 0.0000])
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
            
        # tensor([False, False, False, False,  True, False, False, False, False, False])
        masked_indices = torch.bernoulli(probability_matrix).bool()
            
        # tensor([-100, -100, -100, -100, 4030, -100, -100, -100, -100])
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        #inputs[masked_indices] = self.mask_token_idx
        
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.mask_token_idx
            
        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.bert_tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        
        if idx == 0: self.z1_masked_input = inputs.numpy()
        else: self.z2_masked_input = inputs.numpy()
            
    def masking_collator(self, input_ids):

        for i in range(2):
            try:
                start_padding_idx = (input_ids.squeeze(0) == self.pad_token_idx).nonzero()[0].data.cpu().numpy()[0]
            except IndexError:
                start_padding_idx = self.args.max_len
        
            mask = torch.zeros(start_padding_idx)

            mask_num = int(start_padding_idx * self.args.masking_ratio)
            random_list = [random.randint(0, start_padding_idx-2) for r in range(mask_num)]

            for idx in random_list:
                mask[idx] = 1.0

            mask_input = input_ids.squeeze(0)[:start_padding_idx]
            masked_inputs = mask_input.masked_fill(mask == 1, self.mask_token_idx)

            for idx in range(self.args.max_len - len(masked_inputs)):
                masked_inputs = torch.cat([masked_inputs, torch.tensor([self.pad_token_idx])], dim=-1)
            
            if i == 0: self.z1_masked_input.append(masked_inputs)
            else: self.z2_masked_input.append(masked_inputs)
        
    def __getitem__(self, index):
    
        if self.type == 'train':
    
            inputs = {'anchor': {
                'source': torch.LongTensor(self.input_ids[index]),
                'attention_mask': torch.LongTensor(self.attention_mask[index]),
                'token_type_ids': torch.LongTensor(self.token_type_ids[index]),
                'z1_masked_input': torch.LongTensor(self.z1_masked_input[index]),
                'z2_masked_input': torch.LongTensor(self.z2_masked_input[index]),
                }}
        else:

            inputs = {'sentence_1': {
                'source': torch.LongTensor(self.sentence_1[index]['input_ids']),
                'attention_mask': self.sentence_1[index]['attention_mask'],
                'token_type_ids': torch.LongTensor(self.sentence_1[index]['token_type_ids'])
                },
                      'sentence_2': {
                'source': torch.LongTensor(self.sentence_2[index]['input_ids']),
                'attention_mask': self.sentence_2[index]['attention_mask'],
                'token_type_ids': torch.LongTensor(self.sentence_2[index]['token_type_ids'])
                },
                      'label': {
                          'value': torch.FloatTensor([self.label[index]])}
                }

        for key, value in inputs.items():
            for inner_key, inner_value in value.items():
                inputs[key][inner_key] = inner_value.squeeze(0)
                
        inputs = self.metric.move2device(inputs, self.args.device)
        
        return inputs

    def __len__(self):
        if self.type == 'train':
            return len(self.input_ids)
        else:
            return len(self.label)


# Get train, valid, test data loader and BERT tokenizer
def get_loader(args, metric):
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    path_to_train_data = args.path_to_data + '/' + args.train_data
    path_to_valid_data = args.path_to_data + '/' + args.valid_data
    path_to_test_data = args.path_to_data + '/' + args.test_data

    if args.train == 'True' and args.test == 'False':
        train_iter = ModelDataLoader(path_to_train_data, args, metric, tokenizer, type_='train')
        valid_iter = ModelDataLoader(path_to_valid_data, args, metric, tokenizer, type_='valid')
        
        train_iter.load_data('train')
        valid_iter.load_data('valid')

        loader = {'train': DataLoader(dataset=train_iter,
                                      batch_size=args.batch_size,
                                      shuffle=True),
                  'valid': DataLoader(dataset=valid_iter,
                                      batch_size=args.batch_size,
                                      shuffle=True)}

    elif args.train == 'False' and args.test == 'True':
        test_iter = ModelDataLoader(path_to_test_data, args, metric, tokenizer, type_='test')
        test_iter.load_data('test')

        loader = {'test': DataLoader(dataset=test_iter,
                                     batch_size=args.batch_size,
                                     shuffle=True)}

    else:
        loader = None
    
    return loader, tokenizer


def convert_to_tensor(corpus, tokenizer, device):
    inputs = tokenizer(corpus,
                       truncation=True,
                       return_tensors="pt",
                       max_length=50,
                       pad_to_max_length="right")
    
    embedding = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    token_type_ids = inputs['token_type_ids']
        
    inputs = {'source': torch.LongTensor(embedding).to(device),
              'token_type_ids': torch.LongTensor(token_type_ids).to(device),
              'attention_mask': attention_mask.to(device)}
    
    return inputs


def example_model_setting(model_name):

    from model.simcse.bert import BERT

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = BERT(AutoModel.from_pretrained(model_name))
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.to(device)
    model.eval()
    
    return model, tokenizer, device


if __name__ == '__main__':
    get_loader('test')
