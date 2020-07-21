#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support, roc_curve,roc_auc_score,r2_score,mean_squared_error


# In[2]:


class Kbar(object):
    """Progress bar, taken and adapted from : https://github.com/yueyericardo/pkbar/blob/master/pkbar/pkbar.py"""
    """Keras progress bar.
    Arguments:
            target: Total number of steps expected, None if unknown.
            width: Progress bar width on screen.
            verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
            stateful_metrics: Iterable of string names of metrics that
                    should *not* be averaged over time. Metrics in this list
                    will be displayed as-is. All others will be averaged
                    by the progbar before display.
            interval: Minimum visual progress update interval (in seconds).
            unit_name: Display name for step counts (usually "step" or "sample").
    """

    def __init__(self, target, width=30, verbose=1, interval=0.05,
                 stateful_metrics=None, unit_name='step'):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        self.unit_name = unit_name
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = ((hasattr(sys.stdout, 'isatty')
                                  and sys.stdout.isatty())
                                 or 'ipykernel' in sys.modules
                                 or 'posix' in sys.modules)
        self._total_width = 0
        self._seen_so_far = 0
        # We use a dict + list to avoid garbage collection
        # issues found in OrderedDict
        self._values = {}
        self._values_order = []
        self._start = time.time()
        self._last_update = 0

    def update(self, current, values=None):
        """Updates the progress bar.
        Arguments:
                current: Index of current step.
                values: List of tuples:
                        `(name, value_for_last_step)`.
                        If `name` is in `stateful_metrics`,
                        `value_for_last_step` will be displayed as-is.
                        Else, an average of the metric over time will be displayed.
        """
        values = values or []
        for k, v in values:
            if k not in self._values_order:
                self._values_order.append(k)
            if k not in self.stateful_metrics:
                if k not in self._values:
                    self._values[k] = [v * (current - self._seen_so_far),
                                       current - self._seen_so_far]
                else:
                    self._values[k][0] += v * (current - self._seen_so_far)
                    self._values[k][1] += (current - self._seen_so_far)
            else:
                # Stateful metrics output a numeric value. This representation
                # means "take an average from a single value" but keeps the
                # numeric formatting.
                self._values[k] = [v, 1]
        self._seen_so_far = current

        now = time.time()
        info = ' - %.0fs' % (now - self._start)
        if self.verbose == 1:
            if (now - self._last_update < self.interval
                    and self.target is not None and current < self.target):
                return

            prev_total_width = self._total_width
            if self._dynamic_display:
                sys.stdout.write('\b' * prev_total_width)
                sys.stdout.write('\r')
            else:
                sys.stdout.write('\n')

            if self.target is not None:
                numdigits = int(np.log10(self.target)) + 1
                bar = ('%' + str(numdigits) + 'd/%d [') % (current, self.target)
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % current

            self._total_width = len(bar)
            sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0
            if self.target is not None and current < self.target:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = '%d:%02d:%02d' % (eta // 3600,
                                                   (eta % 3600) // 60,
                                                   eta % 60)
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format
            else:
                if time_per_unit >= 1 or time_per_unit == 0:
                    info += ' %.0fs/%s' % (time_per_unit, self.unit_name)
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/%s' % (time_per_unit * 1e3, self.unit_name)
                else:
                    info += ' %.0fus/%s' % (time_per_unit * 1e6, self.unit_name)

            for k in self._values_order:
                info += ' - %s:' % k
                if isinstance(self._values[k], list):
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            if self.target is not None and current >= self.target:
                info += '\n'

            sys.stdout.write(info)
            sys.stdout.flush()

        elif self.verbose == 2:
            if self.target is not None and current >= self.target:
                numdigits = int(np.log10(self.target)) + 1
                count = ('%' + str(numdigits) + 'd/%d') % (current, self.target)
                info = count + info
                for k in self._values_order:
                    info += ' - %s:' % k
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'

                sys.stdout.write(info)
                sys.stdout.flush()

        self._last_update = now

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)


class Pbar(object):
    """ Progress bar with title and timer
    Arguments:
    name: the bars name.
    target: Total number of steps expected.
    width: Progress bar width on screen.
    Usage example
    ```
    import kpbar
    import time
    pbar = kpbar.Pbar('loading and processing dataset', 10)
    for i in range(10):
        time.sleep(0.1)
        pbar.update(i)
    ```
    ```output
    loading and processing dataset
    10/10  [==============================] - 1.0s
    ```
    """
    def __init__(self, name, target, width=30):
        self.name = name
        self.target = target
        self.start = time.time()
        self.numdigits = int(np.log10(self.target)) + 1
        self.width = width
        print(self.name)

    def update(self, step):

        bar = ('%' + str(self.numdigits) + 'd/%d ') % (step + 1, self.target)

        status = ""

        if step < 0:
            step = 0
            status = "negtive?...\r\n"

        stop = time.time()

        status = '- {:.1f}s'.format((stop - self.start))

        progress = float(step + 1) / self.target

        # prog
        prog_width = int(self.width * progress)
        prog = ''
        if prog_width > 0:
            prog += ('=' * (prog_width - 1))
            if step + 1 < self.target:
                prog += '>'
            else:
                prog += '='
        prog += ('.' * (self.width - prog_width))

        # text = "\r{0} {1} [{2}] {3:.0f}% {4}".format(self.name, bar, prog, pregress, status)

        text = "\r{0} [{1}] {2}".format(bar, prog, status)
        sys.stdout.write(text)
        if step + 1 == self.target:
            sys.stdout.write('\n')
        sys.stdout.flush()


# In[3]:


class SeqDataset(Dataset):
    """
    Class for loading Data
    """

    def __init__(self, seqs_path, genenames_path, expressions_path, transform=None):
        """
        Args :
            seq_path (string) : path to sequences
            tags_path (string) : path to tags
            names_path (string) : path to seq names
            transform (callable, optional) : Optional transform to be applied on sample
        """
        #print("Loading sequences ...")
        temp_seqs = np.load(seqs_path)
        #print("Formating sequences ...")
        temp_seqs = np.resize(temp_seqs, (temp_seqs.shape[0], 1, temp_seqs.shape[1], temp_seqs.shape[2]))
        self.seqs = torch.from_numpy(temp_seqs).float()
        #print("Loading genenames ...")
        self.genenames = np.load(genenames_path)
        #print("Loading expressions ...")
        self.exps = np.load(expressions_path)
        self.transform = transform

    def __len__(self):
        return len(self.exps)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        seq = self.seqs[idx]
        genename = self.genenames[idx]
        exp = self.exps[idx]
        if self.transform:
            seq = self.transform(seq)

        return {'seq':seq, 'exp':exp, 'genename':genename}


# In[4]:


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.
    source :  https://github.com/Bjarten/early-stopping-pytorch/blob/master """
    def __init__(self, path, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = -1
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# In[5]:


class CnnModel(nn.Module):
    def __init__(self):
        super(CnnModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 50, (5, 4), bias=False)
        self.batch1 = nn.BatchNorm2d(50, track_running_stats=False)
        self.maxpool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(50, 30, 3, bias=False)
        #self.conv3 = nn.Conv1d(30, 30, 3, bias=False)
        self.flatt = nn.Flatten()
        self.dense1 = nn.Linear(1380, 500)
        self.dropout1 = nn.Dropout(p=0.3)
        self.dense2 = nn.Linear(500, 200)
        self.dense_output = nn.Linear(200, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.batch1(x)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2])
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        #x = self.conv3(x)
        #x = F.relu(x)
        x = self.flatt(x)
        x = self.dense1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        output = self.dense_output(x)
        return output


# In[6]:


def load_data(path_to_data,batch,split_ratio = 0.2):
    
    """
    General Model
    This function load sequences, expressions and names for the general model and split the training data into a validation loader and
    a train loader 
    :path_to_data : path to data (seqs , genenames and exps numpy files)
    :batch: size of the batch
    :split_ratio: ratio to split the training data into valid and training loader
    """

    seq_train_path = path_to_data + "seqs_trainset.npy"
    exps_train_path = path_to_data + "exps_trainset.npy"
    genenames_train_path = path_to_data + "genenames_trainset.npy"

    seq_test_path = path_to_data + "seqs_testset.npy"
    exps_test_path = path_to_data + "exps_testset.npy"
    genenames_test_path = path_to_data + "genenames_testset.npy"

    
    train_data = SeqDataset(seqs_path=seq_train_path,
                            genenames_path=genenames_train_path,
                            expressions_path=exps_train_path)
    test_data = SeqDataset(seqs_path=seq_test_path,
                           genenames_path=genenames_test_path,
                           expressions_path=exps_test_path)

    # obtain training indices that will be used for validation
    print("train data :", len(train_data))
    print("test data :", len(test_data))
    
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split_number = int(np.floor(split_ratio * num_train))
    train_idx, valid_idx = indices[split_number:], indices[:split_number]

    
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(train_data, shuffle=False, batch_size= batch, sampler=train_sampler)
    valid_loader = DataLoader(train_data, shuffle=False, batch_size= batch, sampler=valid_sampler)
    test_loader = DataLoader(test_data, batch_size= batch)

    return train_loader, valid_loader, test_loader


# In[7]:


def load_clusterdata(path_to_data,batch,cluster,split_ratio = 0.2):
    
    """
    Cluster Model
    This function load sequences, expressions and names for the cluster and split the training data into a validation loader and
    a train loader 
    :path_to_data : path to data (seqs , genenames and exps numpy files)
    :cluster : indice of the cluster
    :batch: size of the batch
    :split_ratio: ratio to split the training data into valid and training loader
    """
    
    seq_train_path = path_to_data + "cluster"+ str(cluster)+"_seqs_trainset.npy"
    exps_train_path = path_to_data + "cluster"+ str(cluster)+"_exps_trainset.npy"
    genenames_train_path = path_to_data + "cluster"+ str(cluster)+"_genenames_trainset.npy"

    seq_test_path = path_to_data + "cluster"+ str(cluster)+"_seqs_testset.npy"
    exps_test_path = path_to_data + "cluster"+ str(cluster)+"_exps_testset.npy"
    genenames_test_path = path_to_data + "cluster"+ str(cluster)+"_genenames_testset.npy"

    train_data = SeqDataset(seqs_path=seq_train_path,
                            genenames_path=genenames_train_path,
                            expressions_path=exps_train_path)
    test_data = SeqDataset(seqs_path=seq_test_path,
                           genenames_path=genenames_test_path,
                           expressions_path=exps_test_path)

    # obtain training indices that will be used for validation
    print("train data :", len(train_data))
    print("test data :", len(test_data))
    
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split_number = int(np.floor(split_ratio * num_train))
    train_idx, valid_idx = indices[split_number:], indices[:split_number]

    
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(train_data, batch_size= batch, sampler=train_sampler)
    valid_loader = DataLoader(train_data, batch_size= batch, sampler=valid_sampler)
    test_loader = DataLoader(test_data, batch_size= batch)

    return train_loader, valid_loader, test_loader


# In[8]:


def load_testdatacluster(cluster, batch):
    
    """
    For load the cluster test loader on an other cluster trained model
    This function load test sequences, expressions and names for the cluster
    :cluster : indice of the cluster to load test data
    :batch: size of the batch
    """
    
    if cluster == 0 :
        path = "./dataset/"
        seq_test_path = path + "seqs_testset.npy"
        exps_test_path = path + "exps_testset.npy"
        genenames_test_path = path + "genenames_testset.npy"

   
        test_data = SeqDataset(seqs_path=seq_test_path,
                               genenames_path=genenames_test_path,
                               expressions_path=exps_test_path)

        test_loader = DataLoader(test_data, batch_size= batch)
        print("test cluster data :",len(test_loader)*batch)

    
    else:
        path = "./cluster_dataset/"
        seq_test_path = path + "cluster"+ str(cluster)+"_seqs_testset.npy"
        exps_test_path = path + "cluster"+ str(cluster)+"_exps_testset.npy"
        genenames_test_path = path + "cluster"+ str(cluster)+"_genenames_testset.npy"


        test_data = SeqDataset(seqs_path=seq_test_path,
                               genenames_path=genenames_test_path,
                               expressions_path=exps_test_path)

        test_loader = DataLoader(test_data, batch_size= batch)
        
        print("test cluster data :",len(test_loader)*batch)

    return test_loader


# In[9]:


def training_pytorch_model(custom_model, train_loader, valid_loader, batch, epoch,
                           criterion, optimizer, device,path, patience=7, keep_track=True):
    """
    This function trains a pytorch model with progress bar and early stopping
    :custom_model: PyTorch model
    :train_loader: PyTorch DataLoader (containing train data)
    :valid_loader: PyTorch DataLoader (containing validation data)
    :batch: batch size
    :epoch: number of epochs
    :criterion: PyTorch Loss function
    :optimizer: Pytorch Optimizer
    :device: Pytorch Device (GPU or CPU)
    :path : path to save model's parameters
    :patience: early stopping patience
    :keep_track: bool (if set to True, will return tracking parameters)
    """
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    # to track outputs

    valid_spear = []
    valid_pear = []

    avg_valid_spear = []
    avg_valid_pear = []
    early_id = 0

    custom_model.to(device)
    early_stopping = EarlyStopping(path,patience=patience, verbose=True)

    for e in range(epoch):

        print(" ")
        print('Epoch: %d/%d' % (e + 1, epoch))
        # progress bar
        kbar = Kbar(target=(len(train_loader) + len(valid_loader))*batch, width=12)

        custom_model.train()  # Sets the model in training mode, which changes the behaviour of dropout layers...

        for i_batch, item in enumerate(train_loader):
            seq = item['seq'].to(device)
            exp = item['exp'].to(device)
                    
            exp = exp.float() 
            
            # Forward pass
            outputs = custom_model(seq)
            
            exp = exp.reshape(exp.shape[0],1)
            #outputs = np.squeeze(outputs)
            #exp = exp.float()
            loss = criterion(outputs, exp)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = loss.item()
            train_losses.append(train_loss)
            
            outputs = outputs.reshape(outputs.shape[0])
            exp = exp.reshape(exp.shape[0])

            spearman = spearmanr(exp.tolist(), outputs.tolist())[0]
            pearson = pearsonr(exp.tolist(), outputs.tolist())[0]
            
            #print("Spearman",spearman)
            #print("Pearson",pearson)

            kbar.update(i_batch * batch, values=[("train_loss", train_loss),("pearson", pearson),("spearman", spearman)])

        custom_model.eval()  # Sets the model in training mode, which changes the behaviour of dropout layers...

        for i_val, item_val in enumerate(valid_loader):
            seq_valid = item_val['seq'].to(device)
            exp_valid = item_val['exp'].to(device)
            
            exp_valid = exp_valid.float() 

            outputs_valid = custom_model(seq_valid)
            #exp_valid = exp_valid.view(-1, 1)
            
            exp_valid = exp_valid.reshape(exp_valid.shape[0],1)
            #exp_valid = exp.float()

            loss_v = criterion(outputs_valid, exp_valid)

            outputs_valid = outputs_valid.reshape(outputs_valid.shape[0])
            exp_valid = exp_valid.reshape(exp_valid.shape[0])
            
            spearman = spearmanr(exp_valid.tolist(), outputs_valid.tolist())[0]
            pearson = pearsonr(exp_valid.tolist(), outputs_valid.tolist())[0]

            valid_spear.append(spearman)
            valid_pear.append(pearson)

            valid_loss = loss_v.item()
            valid_losses.append(valid_loss)

            values = [("valid_loss", valid_loss),("pearson_valid", pearson), ("spearman_valid", spearman)]
            kbar.update((i_val + i_batch) * batch, values=values)

        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        valid_pearson = np.average(valid_pear)
        valid_spearman = np.average(valid_spear)

        avg_valid_pear.append(valid_pearson)
        avg_valid_spear.append(valid_spearman)

        train_losses = []
        valid_losses = []
        valid_spear = []
        valid_pear = []

        print(" ")

        early_stopping(valid_loss, custom_model)

        if early_stopping.early_stop:
            early_id = e - patience
            print(" Early Stopping ")
            break

        print_msg = (f'train_loss: {train_loss:.5f} ' + f'average_valid_loss: {valid_loss:.5f} '
                     + f'average pearson corr: {avg_valid_pear[-1]:.5f} '
                     + f'average spearman corr: {avg_valid_spear[-1]:.5f}')
        print("")
        print(print_msg)

    custom_model.load_state_dict(torch.load(path))

    if keep_track is True:
        return avg_train_losses, avg_valid_losses, avg_valid_spear, avg_valid_pear, early_id
    else :
        return 0


# In[10]:


def test_pytorch_model(custom_model, test_loader, device,path_save, verbose=True):
    """
    This function tests a PyTorch model.
    :custom_model: PyTorch model
    :test_loader: PyTorch test dataLoader (contains testing set)
    :device: PyTorch device (GPU or CPU)
    :path_save : path to load model's parameters
    :verbose: bool (if set to True, will display Spearman and Pearson correlations).
    """
    # initialize lists to monitor test loss and accuracy
    preds = []
    true = []
    custom_model.load_state_dict(torch.load(path_save))


    custom_model.train(False)  # prep model for evaluation

    for item in test_loader:
        seq = item['seq'].to(device)
        exp = item['exp'].to(device)

        exp = exp.view(-1, 1)

        # forward pass: compute predicted outputs by passing inputs to the model
        output = custom_model(seq)
        
        output = output.reshape(output.shape[0])
        exp = exp.reshape(exp.shape[0])
        
        true += exp.tolist()
        preds += output.tolist()

    if verbose is True:
        print("Spearman correlation: ", spearmanr(preds, true)[0])
        print("Pearson correlation: ", pearsonr(preds,true)[0])
        
    r2_s = r2_score(true, preds)
    mse =  mean_squared_error(true, preds)
    print(r2_score)
    print(mse)
   

    return preds, true


# In[11]:


def plot_loss_curve(avg_train_losses, avg_valid_losses,indice,earlyid,batch_size,epoch,spearman_test,pearson_test):
    
    """
    This function plot and saves loss curves.
    :avg_train_losses: list (list of avg train loss per epoch)
    :avg_valid_losses: list (list of avg valid loss per epoch)
    :indice : indice of the cluster
    :earlyid: int (if not set to None, will show a line showing early stopping point)
    :batch_size: size of the batch
    :epoch: number of epochs    
    :spearman_test: value of the spearman correlation for the cluster
    :pearson_test: value of the pearson correlation for the cluster
    """
    
    if indice == 0 :
        imgname = 'img_LossModel'+ '_e'+ str(epoch)+'_b'+ str(batch_size)
    else : 
        imgname = 'img_Losscluster'+ str(indice) + '_e'+ str(epoch)+'_b'+ str(batch_size)
    
    fig, axs = plt.subplots(figsize=(10, 8))
    
    # visualize the loss as the network trained
    plt.plot(range(1, len(avg_train_losses) + 1), avg_train_losses, label='Training Loss')
    plt.plot(range(1, len(avg_valid_losses) + 1), avg_valid_losses, label='Validation Loss')

    if earlyid !=0:
        plt.axvline(earlyid, linestyle='--', color='r', label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(min(avg_valid_losses) - 0.1 * min(avg_valid_losses),
             max(avg_valid_losses) + 0.1 * max(avg_valid_losses))  # consistent scale
    plt.xlim(0, len(avg_train_losses) + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    axs.set_title(imgname)

    text = "Cluster " + str(indice) + "\nSpearman : " + str(spearman_test) + "\n Pearson : " + str(pearson_test) 
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    axs.text(0.05, 0.95, text, transform=axs.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
    
    plt.show()
    
    path = "./results/" + imgname
    fig.savefig(path, bbox_inches='tight')
    
    


# In[12]:


def plot_corr_curve(avg_valid_pear, avg_valid_spear,indice,earlyid,batch_size,epoch,spearman_test,pearson_test):
    """
    This function saves/plot loss curves.
    :avg_valid_pear: list (list of avg valid Pearson correlation per epoch)
    :avg_valid_spear: list (list of avg valid Spearman correlation per epoch
    :indice : indice of the cluster
    :earlyid: int (if not set to None, will show a line showing early stopping point)
    :batch_size: size of the batch
    :epoch: number of epochs    
    :spearman_test: value of the spearman correlation for the cluster
    :pearson_test: value of the pearson correlation for the cluster
    """
    
    
    if indice == 0 :
        imgname = 'img_CorrModel'+ '_e'+ str(epoch)+'_b'+ str(batch_size)
    else : 
        imgname = 'img_Corrcluster'+ str(indice) + '_e'+ str(epoch)+'_b'+ str(batch_size)
        
   
    fig, axs = plt.subplots(figsize=(10, 8))
    # visualize the loss as the network trained
    plt.plot(range(1, len(avg_valid_pear) + 1), avg_valid_pear, label='Validation Pearson Correlation')
    plt.plot(range(1, len(avg_valid_spear) + 1), avg_valid_spear, label='Validation Spearman Correlation')

    if earlyid != 0:
        plt.axvline(earlyid, linestyle='--', color='r', label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('correlation')
    plt.ylim(0, 1)  # consistent scale
    plt.xlim(0, len(avg_valid_spear) + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    axs.set_title(imgname)

    text = "Cluster " + str(indice) + "\nSpearman : " + str(spearman_test) + "\n Pearson : " + str(pearson_test) 
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    axs.text(0.05, 0.95, text, transform=axs.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
    
    plt.show()
    
    path = "./results/" + imgname
    fig.savefig(path, bbox_inches='tight')  


# In[15]:


def write_listtransferresults(cluster,cluster_len,spearmantest_list,pearsontest_list,batch_size,epoch,text) : 
    path = "./resultstxt/"+text+"_e"+ str(epoch)+"_b"+ str(batch_size) + "_" + "result_testcluster"+str(cluster)
    f = open(path, "w")
    for i in range(cluster_len):
        if  (pearsontest_list[i]>0.5) :
            print("Test cluster " + str(cluster) + " -> cluster " + str(i) + ": pearson = " + str(pearsontest_list[i])+ "\n")
        if (spearmantest_list[i]>0.5) :
            print("Test cluster " + str(cluster) + " -> cluster " + str(i) + ": spearman = "+ str(spearmantest_list[i]) + "\n")

        f.write("Test cluster " + str(cluster) + " -> cluster " + str(i) + ": spearman = "+ str(spearmantest_list[i]) + " pearson = " + str(pearsontest_list[i])+ "\n")
    f.close    


# In[16]:


def write_resultfile(path,cluster,cluster_len,spearmantest_list,pearsontest_list) : 
    """
    This function saves cluster's spearman and pearson test for one cluster .
    :path: path to save results 
    :cluster: indice of the cluster
    :cluster_len : number of clusters
    :spearmantest_list: list of spearman tests values for the cluster 
    :pearsontest_list: list of pearson tests values for the cluster 
    """
    
    f = open(path, "a")
    f.write("\n")
    f.write("Cluster"+str(cluster)+" : \n")
    for i in range(cluster_len):
        if  (pearsontest_list[i]>0.5) :
            print(" ||||| Test cluster " + str(cluster) + " -> cluster " + str(i) + ": pearson = " + str(pearsontest_list[i])+ " ||||| \n")
        if (spearmantest_list[i]>0.5) :
            print(" ||||| Test cluster " + str(cluster) + " -> cluster " + str(i) + ": spearman = "+ str(spearmantest_list[i]) + " ||||| \n")

        f.write("Test cluster " + str(cluster) + " -> cluster " + str(i) + ": spearman = "+ str(spearmantest_list[i]) + " pearson = " + str(pearsontest_list[i])+ "\n")
    f.close    


# In[17]:


clusters_spearman = []
clusters_pearson = []
for i in range(21):
    print("\n***** Démarrage entrainement/test avec le cluster",i,"*****\n")

    #paramètres du modèle
    text = "2CONV_log"
    epoch = 300
    net = CnnModel()
    #learning_rate = 0.00001
    #optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(net.parameters())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 256
    if(i == 0) :
        #modèle général
        #paramètres du modèle
        batch_sizec0 = 700
        path = "./dataset/"
        train_loader, valid_loader, test_loader = load_data(path,batch_sizec0)
        path_save = "CNNModel_checkpoint.pt"
        print("Save at",path_save)
        avg_train_losses, avg_valid_losses, avg_valid_spear, avg_valid_pear, early_id = training_pytorch_model(net, train_loader, valid_loader, batch_sizec0, epoch, criterion, optimizer, device,path_save)

    else :
        #modèle spécifique
        path_save = "CNNModelcluster"+str(i)+"_checkpoint.pt"
        print("Save at",path_save)
        path = "./cluster_dataset/"
        train_loader, valid_loader, test_loader = load_clusterdata(path,batch_size,i)
        avg_train_losses, avg_valid_losses, avg_valid_spear, avg_valid_pear, early_id = training_pytorch_model(net, train_loader, valid_loader, batch_size, epoch, criterion, optimizer, device,path_save)
    
    print("\n*** Entrainement cluster",i,"terminé ***\n")

    spearmantest_list = []
    pearsontest_list = []
    print("model save at",path_save)
    for j in range(21) :
        
        print("\n--- Test transfer avec cluster",j,"---\n")
        test_loader = load_testdatacluster(j, batch_size)
        preds, true = test_pytorch_model(net, test_loader, device,path_save)
        spearman_test = spearmanr(preds, true)[0]
        pearson_test = pearsonr(preds,true)[0]
        
        spearmantest_list.append(spearman_test)
        pearsontest_list.append(pearson_test)        
        
        #write_transferresults(i,j,spearman_test,pearson_test)
        
    path_file = "./resultstxt/"+text+"_e"+ str(epoch)+"_bc0"+ str(batch_sizec0)+"_bcall"+str(batch_size)
    write_resultfile(path_file,i,21,spearmantest_list,pearsontest_list)

    #write_listtransferresults(i,21,spearmantest_list,pearsontest_list,batch_size,epoch,text)

    print("\n***** Fin entrainement/test avec le cluster",i,"*****\n")

