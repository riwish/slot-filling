import os
import sys
import string
import pickle
import random
import argparse
import logging
import numpy as np

from itertools import chain

logger = logging.getLogger(__name__)

def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)



def load_knowledge(args):
    return load_pickle(os.path.join(args.data_dir, 'knowledge.pickle'))


def load_pickle(filename):
    with open(os.path.join(filename), 'rb') as f:
        data = pickle.load(f)
    return data

def save_pickle(data, filename):
    with open(os.path.join(filename), 'wb') as f:
        pickle.dump(data, f)

def convert_int(arr):
    """
    Tries to cast object to int
    :param arr:
    :return:
    """
    try:
        a = int(arr)
    except:
        return None
    return a


def shuffle_split(X, y, percentage):
    arr_rand = np.random.random(X.shape[0])
    split = arr_rand < np.percentile(arr_rand, 100-percentage)
    
    X_train = X[split]
    y_train = y[split]
    X_test = X[~split]
    y_test = y[~split]
    return X_train, X_test, y_train, y_test


def check_validity(X_train, X_test):
    # Check if any duplicate dp from train resides in test set
    for dp in X_test:
        assert dp not in X_train

        
def remove_duplicates(X, y):
    data = dict(zip(X, y))
    # Check if lenghts between target and features are equal
    for k, v in data.items():
        assert len(k.split(' ')) == len(v.split(' '))
    X = np.asarray(list(data.keys()))
    y = np.asarray(list(data.values()))
    return X, y


def read_raw_seq_data(args):
    X = []
    y = []
    total_datapoints = 0
    with open(os.path.join(args.data_dir, args.task , 'raw', 'seq.in'), 'r', encoding='utf-8') as f_in, open(os.path.join(args.data_dir, args.task, 'raw', 'seq.out'), 'r', encoding='utf-8') as f_out:
        for line_in, line_out in zip(f_in, f_out):
            if len(line_in)>1:
                X.append(line_in.strip())
                y.append(line_out.strip())
                total_datapoints += 1
    print("Total RAW samples => ", total_datapoints)
    X, y = remove_duplicates(X, y)
    return X, y


def print_subset_lengths(X_train, X_val, X_test, y_train, y_val, y_test):
    print("X_train => ", len(X_train))
    print("X_val => ", len(X_val))
    print("X_test => ", len(X_test))

    print("y_train => ", len(y_train))
    print("y_val => ", len(y_val))
    print("y_test => ", len(y_test))


def write_to_file(args, data):
    # First create directories if not present already
    for sub_dir in data.keys():
        if not os.path.exists(os.path.join(args.data_dir, args.task, sub_dir)):
            os.makedirs(os.path.join(args.data_dir, args.task, sub_dir))
    # Write the subsets to a file
    for sub_dir, dataset in data.items():
        seq_in = open(os.path.join(args.data_dir, args.task, sub_dir, 'seq.in'), 'w', encoding='utf-8')
        seq_out = open(os.path.join(args.data_dir, args.task, sub_dir, 'seq.out'), 'w')
        for X, y in zip(dataset[0], dataset[1]):
            seq_in.write(X + '\n')
            seq_out.write(y + '\n')
        seq_in.close()
        seq_out.close()
    

def read_processed_dataset(args, dataset):
    for subdir, data in dataset.items():  
        with open(os.path.join(args.data_dir, args.task, subdir, 'seq.in'), 'r', encoding='utf-8') as seq_in, open(os.path.join(args.data_dir, args.task, subdir, 'seq.out'), 'r', encoding='utf-8') as seq_out:
            for line_in, line_out in zip(seq_in, seq_out):
                dataset[subdir][0].append(line_in.strip())
                dataset[subdir][1].append(line_out.strip())
        print(len(dataset[subdir][0]))
    return dataset


def mask_values(args, dataset):
    logging.info("Performing MASKING only")
    # Start masking procedure
    for name, data in dataset.items():
        for sen_idx, (seq_in, seq_out) in enumerate(zip(data[0], data[1])):
            seq_in = seq_in.split(' ')
            seq_out = seq_out.split(' ')
            for word_idx, (w_in, w_out) in enumerate(zip(seq_in, seq_out)):
                # Introduce a random chance replacing threshold
                chance = random.randint(1, 100)
                if w_out not in ['O', 'PAD', 'UNK', 'MASK']:    
                    # Mask all slot related annotations
                    seq_in[word_idx] = 'MASK'
                # Randomly mask words by a defined percentage
                if chance <= args.mask_percentage:
                    seq_in[word_idx] = 'MASK'
            dataset[name][0][sen_idx] = ' '.join(seq_in)
            assert len(dataset[name][0][sen_idx].split(' ')) == len(dataset[name][1][sen_idx].split(' '))

    # Check result
    print(dataset['train'][0][:3])
    print(dataset['train'][1][:3])
    return dataset


def augment_values(args, dataset):
    logging.info("Performing AUGMENTATION only")
    # load augmentation db and keep matching keys
    augmentation_db = load_knowledge(args)
    to_drop = list(set(augmentation_db.keys() - set(args.augmentation_list)))
    for d in to_drop:
        del augmentation_db[d]
    augmentation_db['per.city'] = augmentation_db['inc.city']
    
    # Start masking procedure
    for name, data in dataset.items():
        for sen_idx, (seq_in, seq_out) in enumerate(zip(data[0], data[1])):
            seq_in = seq_in.split(' ')
            seq_out = seq_out.split(' ')
            for word_idx, (w_in, w_out) in enumerate(zip(seq_in, seq_out)):
                if w_out not in ['O', 'PAD', 'UNK', 'MASK']:    
                    # Perform data augmentation with a 25% chance whenever augmented data is available
                    if w_out in args.augmentation_list:
                        aug_chance = random.randint(1, 100)
                        if aug_chance <= 50:
                            # Augment this value - select random value from db)
                            new_value = random.choice(augmentation_db[w_out]['text']).strip().translate(str.maketrans('', '', string.punctuation))
                            try:
                                v = new_value.split(' ')
                                new_value = v[len(v)-1].strip()
                                if len(new_value) >= 4:
                                    #print(True, seq_in[word_idx], repr(new_value))
                                    seq_in[word_idx] = new_value
                                else:
                                    seq_in[word_idx] = seq_in[word_idx]
                            except:
                                seq_in[word_idx] = new_value
            dataset[name][0][sen_idx] = ' '.join(seq_in)
            assert len(dataset[name][0][sen_idx].split(' ')) == len(dataset[name][1][sen_idx].split(' '))
    return dataset


def mask_with_augmentation(args, dataset):
    logging.info("Performing MASK with AUGMENTATION")
    # load augmentation db and keep matching keys
    augmentation_db = load_knowledge(args)
    to_drop = list(set(augmentation_db.keys() - set(args.augmentation_list)))
    for d in to_drop:
        del augmentation_db[d]
    augmentation_db['per.city'] = augmentation_db['inc.city']

    # Start masking procedure
    for name, data in dataset.items():
        for sen_idx, (seq_in, seq_out) in enumerate(zip(data[0], data[1])):
            #print(sen_idx, seq_in, seq_out)
            seq_in = seq_in.split(' ')
            seq_out = seq_out.split(' ')
            for word_idx, (w_in, w_out) in enumerate(zip(seq_in, seq_out)):
                # Introduce a random chance replacing threshold
                chance = random.randint(1, 100)
                if w_out not in ['O', 'PAD', 'UNK', 'MASK']:    
                    if args.do_augmentation:
                        # Perform data augmentation with a 25% chance whenever augmented data is available
                        if w_out in args.augmentation_list:
                            aug_chance = random.randint(1, 100)
                            if aug_chance <= 50:
                                # Augment this value - select random value from db)
                                new_value = random.choice(augmentation_db[w_out]['text']).strip().translate(str.maketrans('', '', string.punctuation))
                                try:
                                    v = new_value.split(' ')
                                    new_value = v[len(v)-1].strip()
                                    if len(new_value) >= 4:
                                        #print(True, seq_in[word_idx], repr(new_value))
                                        seq_in[word_idx] = new_value
                                    else:
                                        seq_in[word_idx] = seq_in[word_idx]
                                        #print(True, repr(new_value), seq_in[word_idx])
                                except:
                                    #print(repr(new_value))
                                    seq_in[word_idx] = new_value
                            else:
                                # Randomness didnt select augmentation, perform regular masking instead
                                seq_in[word_idx] = 'MASK'
                        # Perform regular masking if not in list
                        else:
                             seq_in[word_idx] = 'MASK'
                    else:
                        # Mask all slot related annotation if no augmentation is performed
                        seq_in[word_idx] = 'MASK'
                # Randomly mask words by a defined percentage
                if chance <= args.mask_percentage:
                    seq_in[word_idx] = 'MASK'
            dataset[name][0][sen_idx] = ' '.join(seq_in)
            assert len(dataset[name][0][sen_idx].split(' ')) == len(dataset[name][1][sen_idx].split(' '))
    return dataset


def create_slot_vocab(args):
    """
    Collects all unique slot labels from the training set and writes it to slot_label.txt
    :param data_dir:
    :return:
    """
    slot_label_vocab = 'slot_label.txt'
    train_dir = os.path.join(args.data_dir, args.task, 'train')

    # Create slot vocabulary
    with open(os.path.join(train_dir, 'seq.out'), 'r', encoding='utf-8') as f_r, open(os.path.join(args.data_dir, args.task, slot_label_vocab), 'w',
                                                                                      encoding='utf-8') as f_w:
        slot_vocab = set()
        for line in f_r:
            line = line.strip()
            slots = line.split()
            for slot in slots:
                slot_vocab.add(slot)
        slot_vocab = sorted(list(slot_vocab), key=lambda x: (x[2:], x[:2]))
        print(slot_vocab)

        # Write additional tokens
        additional_tokens = ["PAD", "UNK", "MASK"]
        for token in additional_tokens:
            f_w.write(token + '\n')

        for slot in slot_vocab:
            f_w.write(slot + '\n')


def create_dataset(args):
    # Read data to numpy arrays and separate target from features
    X, y = read_raw_seq_data(args)
    # Create subsets
    X_train, X_test, y_train, y_test = shuffle_split(X, y, percentage=20)
    X_train, X_val, y_train, y_val = shuffle_split(X_train, y_train, percentage=20)
    # We wont need original np arrays anymore
    del X, y
    X_train = X_train.tolist()
    X_val = X_val.tolist()
    X_test = X_test.tolist()
    y_train = y_train.tolist()
    y_val = y_val.tolist()
    y_test = y_test.tolist()
    # Print lenghts
    print_subset_lengths(X_train, X_val, X_test, y_train, y_val, y_test)
    # Write to file
    data = {'train':[X_train, y_train], 'val':[X_val, y_val], 'test':[X_test, y_test]}
    write_to_file(args, data)
    del X_train, X_val, X_test, y_train, y_val, y_test


def impute_values(args):
    # Define placeholder for data subsets
    dataset = {'train':[[], []]}
    dataset = read_processed_dataset(args, dataset)

    # Implement following mask techniques if applicable
    if args.do_mask and args.do_augmentation:
        mask_with_augmentation(args, dataset)
    elif args.do_mask:
        mask_values(args, dataset)
    elif args.do_augmentation:
        augment_values(args, dataset)
    write_to_file(args, dataset)


def main(args):
    init_logger()
    if args.do_create_dataset:
        logger.info("*** Creating data subsets ***")
        create_dataset(args)
        logger.info("*** Finished creating data subsets ***")
    if args.do_create_slot_vocab:
        logger.info("*** Creating slot vocabulary ***")
        create_slot_vocab(args)
        logger.info("*** Finished creating slot vocabulary ***")
    if args.do_mask or args.do_augmentation:
        logger.info("*** Imputing values in the training data ***")
        impute_values(args)
        logger.info("*** Finished masking the training data ***")

if __name__ != "main":
    DATA_DIR = os.path.normpath(r"PROVIDE ABSOLUTE PATH TO DATA")
    parser = argparse.ArgumentParser()
    # Main arguments
    parser.add_argument("--task", default=None, required=True, type=str, help="The name of the task")
    parser.add_argument("--data_dir", default=DATA_DIR, required=False, type=str, help="The input data dir")
    # Actions
    parser.add_argument("--do_create_slot_vocab", action="store_true", help="Whether to create a slot vocabulary")
    parser.add_argument("--do_create_dataset", action="store_true", help="Whether to subsets from raw ATIS like data.")
    parser.add_argument("--do_mask", action="store_true", help="Whether to mask values in the trainingset.")
    parser.add_argument("--do_augmentation", action="store_true", help="Whether to perform data augmentation besides masking values")
    parser.add_argument("--mask_percentage", default=10, type=int, help="Percentage of values to be masked.")

    args = parser.parse_args()
    args.augmentation_list = ["Define all non artificial slot labels as a list"]
    main(args)
