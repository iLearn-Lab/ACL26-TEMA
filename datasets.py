"""Provides data for training and testing."""
import os
import numpy as np
import PIL
import torch
import json
import torch.utils.data
import string 
import glob
import torchvision
import random
import pickle
from tqdm import trange
def save_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


class M_FashionIQ(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        super().__init__()

        self.path = path
        self.image_dir = self.path + 'resized_image'
        self.split_dir = self.path + 'image_splits'
        self.caption_dir = self.path + 'captions'
        self.transform = transform
        if not os.path.exists(os.path.join(self.path, 'm_fashion_iq_data.json')):
            self.mfashioniq_data = []
            self.train_init_process()
            with open(os.path.join(self.path, 'm_fashion_iq_data.json'), 'w') as f:
                json.dump(self.mfashioniq_data, f)

            self.test_queries_dress, self.test_targets_dress = self.get_test_data('dress')
            self.test_queries_shirt, self.test_targets_shirt = self.get_test_data('shirt')
            self.test_queries_toptee, self.test_targets_toptee = self.get_test_data('toptee')
            save_obj(self.test_queries_dress, os.path.join(self.path, 'test_queries_dress_mmt.pkl'))
            save_obj(self.test_targets_dress, os.path.join(self.path, 'test_targets_dress_mmt.pkl'))
            save_obj(self.test_queries_shirt, os.path.join(self.path, 'test_queries_shirt_mmt.pkl'))
            save_obj(self.test_targets_shirt, os.path.join(self.path, 'test_targets_shirt_mmt.pkl'))
            save_obj(self.test_queries_toptee, os.path.join(self.path, 'test_queries_toptee_mmt.pkl'))
            save_obj(self.test_targets_toptee, os.path.join(self.path, 'test_targets_toptee_mmt.pkl'))

        else:
            with open(os.path.join(self.path, 'm_fashion_iq_data.json'), 'r') as f:
                self.mfashioniq_data = json.load(f) 
            self.test_queries_dress = load_obj(os.path.join(self.path, 'test_queries_dress_mmt.pkl'))
            self.test_targets_dress = load_obj(os.path.join(self.path, 'test_targets_dress_mmt.pkl'))
            self.test_queries_shirt = load_obj(os.path.join(self.path, 'test_queries_shirt_mmt.pkl'))
            self.test_targets_shirt = load_obj(os.path.join(self.path, 'test_targets_shirt_mmt.pkl'))
            self.test_queries_toptee = load_obj(os.path.join(self.path, 'test_queries_toptee_mmt.pkl'))
            self.test_targets_toptee = load_obj(os.path.join(self.path, 'test_targets_toptee_mmt.pkl'))

    def train_init_process(self):
        for name in ['dress', 'shirt', 'toptee']:
            with open(os.path.join(self.caption_dir, "cap.{}.{}.mmt.json".format(name, 'train')), 'r') as f:
                ref_captions = json.load(f)

            for triplets in ref_captions:
                ref_id = triplets['candidate']
                tag_id = triplets['target']
                cap = triplets['mmt']

                self.mfashioniq_data.append({
                    'target': name + '_' + tag_id,
                    'candidate': name + '_' + ref_id,
                    'mmt': cap,
                })


    def __len__(self):
        return len(self.mfashioniq_data)

    def __getitem__(self, idx):
        caption = self.mfashioniq_data[idx]

        mod_str = caption['mmt']
        candidate = caption['candidate']
        target = caption['target']

        out = {}
        out['source_img_data'] = self.get_img(candidate)
        out['target_img_data'] = self.get_img(target)
        out['mod'] = {'str': mod_str}

        return out

    def get_img(self,image_name):
        img_path = os.path.join(self.image_dir, image_name.split('_')[0], image_name.split('_')[1] + ".jpg")
        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f)
            img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)
        return img

    def get_test_img(self,image_name):
        img_path = os.path.join(self.image_dir, image_name.split('_')[0], image_name.split('_')[1] + ".jpg")
        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f)
            img = img.convert('RGB')
        img = self.transform(img)

        return img

    def get_all_texts(self):
        texts = []
        for caption in self.mfashioniq_data:
            mod_texts = caption['mmt']
            texts.append(mod_texts)
        return texts

    def get_test_data(self, name):       # query

        with open(os.path.join(self.split_dir, "split.{}.{}.json".format(name, 'val')), 'r') as f:
            images = json.load(f)
        with open(os.path.join(self.caption_dir, "cap.{}.{}.mmt.json".format(name, 'val')), 'r') as f:
            ref_captions = json.load(f)

        test_queries = []
        for idx in range(len(ref_captions)):
            caption = ref_captions[idx]
            mod_str = caption['mmt']
            candidate = caption['candidate']
            target = caption['target']
            out = {}
            out['source_img_id'] = images.index(candidate)
            out['source_img_data'] = self.get_test_img(name + '_' + candidate)
            out['target_img_id'] = images.index(target)
            out['target_img_data'] = self.get_test_img(name + '_' + target)
            out['mod'] = {'str': mod_str}

            test_queries.append(out)
        
        test_targets_id = []
        for i in test_queries:
            if i['source_img_id'] not in test_targets_id:
                test_targets_id.append(i['source_img_id'])
            if i['target_img_id'] not in test_targets_id:
                test_targets_id.append(i['target_img_id'])
        test_targets = []
        for i in test_targets_id:
            out = {}
            out['target_img_id'] = i
            out['target_img_data'] = self.get_test_img(name + '_' + images[i])      
            test_targets.append(out)
        return test_queries, test_targets


class M_CIRR(torch.utils.data.Dataset):
    def __init__(self, path, transform=None, case_look=False) -> None:
        super(M_CIRR, self).__init__()
        self.path = path
        self.caption_dir = self.path + 'captions'
        self.split_dir = self.path + 'image_splits'
        self.transform = transform
        self.case_look = case_look
        # train data
        with open(os.path.join(self.caption_dir, "cap.rc2.train.mmt.json"), 'r') as f:
            self.cirr_data = json.load(f)

        with open(os.path.join(self.split_dir, "split.rc2.train.json"), 'r') as f:
            self.train_image_path = json.load(f)
            self.train_image_name = list(self.train_image_path.keys())

        with open(os.path.join(self.caption_dir, "cap.rc2.val.mmt.json"), 'r') as f:
            self.val_data = json.load(f)

        with open(os.path.join(self.split_dir, "split.rc2.val.json"), 'r') as f:
            self.val_image_path = json.load(f)
            self.val_image_name = list(self.val_image_path.keys())
        # val data
        if not os.path.exists(os.path.join(self.path, 'mcirr_val_queries.pkl')):
            self.val_queries, self.val_targets = self.get_val_queries()
            save_obj(self.val_queries, os.path.join(self.path, 'mcirr_val_queries.pkl'))
            save_obj(self.val_targets, os.path.join(self.path, 'mcirr_val_targets.pkl'))
        else:
            self.val_queries = load_obj(os.path.join(self.path, 'mcirr_val_queries.pkl'))
            self.val_targets = load_obj(os.path.join(self.path, 'mcirr_val_targets.pkl'))
        # test data
        if not os.path.exists(os.path.join(self.path, 'mcirr_test_queries.pkl')):
            self.test_name_list, self.test_img_data, self.test_queries = self.get_test_queries()
            save_obj(self.test_name_list, os.path.join(self.path, 'mcirr_test_name_list.pkl'))
            save_obj(self.test_img_data, os.path.join(self.path, 'mcirr_test_img_data.pkl'))
            save_obj(self.test_queries, os.path.join(self.path, 'mcirr_test_queries.pkl'))
        else:
            self.test_name_list = load_obj(os.path.join(self.path, 'mcirr_test_name_list.pkl'))
            self.test_img_data = load_obj(os.path.join(self.path, 'mcirr_test_img_data.pkl'))
            self.test_queries = load_obj(os.path.join(self.path, 'mcirr_test_queries.pkl'))

    def __len__(self):
        return len(self.cirr_data)

    def __getitem__(self, idx):
        caption = self.cirr_data[idx]
        reference_name = caption['reference']
        mod_str = caption['mmt']
        target_name = caption['target_hard']

        out = {}
        out['source_img_data'] = self.get_img(self.train_image_path[reference_name])
        out['target_img_data'] = self.get_img(self.train_image_path[target_name])
        out['mod'] = {'str':mod_str}
        
        return out

    def get_img(self, img_path, return_raw=False):
        img_path = os.path.join(self.path, img_path.lstrip('./'))
        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f)
            img = img.convert('RGB')
            
        if return_raw:
            transform = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])
            return transform(img)

        if self.transform:
            img = self.transform(img)
        return img

    def get_val_queries(self):
        with open(os.path.join(self.caption_dir, "cap.rc2.val.mmt.json"), 'r') as f:
            val_data = json.load(f)

        with open(os.path.join(self.split_dir, "split.rc2.val.json"), 'r') as f:
            val_image_path = json.load(f)
            val_image_name = list(val_image_path.keys())
        
        test_queries = []
        for idx in range(len(val_data)):
            caption = val_data[idx]
            mod_str = caption['mmt']
            reference_name = caption['reference']
            target_name = caption['target_hard']
            subset_names = caption['img_set']['members']
            subset_ids = [val_image_name.index(n) for n in subset_names]

            out = {}
            out['source_img_id'] = val_image_name.index(reference_name)
            out['source_img_data'] = self.get_img(val_image_path[reference_name])
            out['target_img_id'] = val_image_name.index(target_name)
            out['target_img_data'] = self.get_img(val_image_path[target_name])
            out['mod'] = {'str':mod_str}
            out['subset_id'] = subset_ids
            
            if self.case_look:
                out['raw_src_img_data'] = self.get_img(val_image_path[reference_name], return_raw=True)
                out['raw_tag_img_data'] = self.get_img(val_image_path[target_name], return_raw=True)
            
            test_queries.append(out)

        test_targets = []
        for i in range(len(val_image_name)):
            name = val_image_name[i]
            out = {}
            out['target_img_id'] = i
            out['target_img_data'] = self.get_img(val_image_path[name])
            if self.case_look:
                out['raw_tag_img_data'] = self.get_img(val_image_path[name], return_raw=True)
            test_targets.append(out)

        return test_queries, test_targets
    
    def get_test_queries(self):

        with open(os.path.join(self.caption_dir, "cap.rc2.test1.json"), 'r') as f:
            test_data = json.load(f)

        with open(os.path.join(self.split_dir, "split.rc2.test1.json"), 'r') as f:
            test_image_path = json.load(f)
            test_image_name = list(test_image_path.keys())

        queries = []
        for i in range(len(test_data)):
            out = {}
            caption = test_data[i]
            out['pairid'] = caption['pairid']
            out['reference_data'] = self.get_img(test_image_path[caption['reference']])
            out['reference_name'] = caption['reference']
            out['mod'] = caption['caption']
            out['subset'] = caption['img_set']['members']
            queries.append(out)

        image_name = []
        image_data = []
        for i in range(len(test_image_name)):
            name = test_image_name[i]
            data = self.get_img(test_image_path[name])
            image_name.append(name)
            image_data.append(data)
        return image_name, image_data, queries

