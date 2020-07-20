#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support, roc_curve,roc_auc_score,RocCurveDisplay


# In[4]:


class SeqDataset(Dataset):
    """
    Class to create the training dataset, validation dataset and test dataset    
    """

    def __init__(self, seq_path, genenames_path, signals_path, transform=None):

        """
        Args :
            seq_path (string) : path to sequences
            genenames_path (string) : path to gene names
            signal_paths (string) : path to signals
            transform (callable, optional) : Optional transform to be applied on sample
        """            
        #print("Loading genes names...")
        self.genenames = np.load(genenames_path)
        #print("Loading signals...")
        self.signals = np.load(signals_path)
        #print("Loading sequences...")
        #self.seqs = np.load(seq_path, allow_pickle=True)
        temp_seqs = np.load(seq_path, allow_pickle=True)
        #print(temp_seqs.shape)

        #print("Formating sequences ...")
        temp_seqs = np.resize(temp_seqs, (temp_seqs.shape[0], 1, temp_seqs.shape[1], temp_seqs.shape[2]))
        self.seqs = torch.from_numpy(temp_seqs).float()
        print("Shape :",self.seqs.shape)
        self.transform = transform
    def __len__(self):
        return len(self.signals)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        seq = self.seqs[idx]
        genename = self.genenames[idx]
        signal = self.signals[idx]
        if self.transform:
            seq = self.transform(seq)
        
        return {'seq':seq, 'signal':signal, 'genename':genename}


# In[5]:


class CnnModel(nn.Module):
    """
    Class for defining the neural network architecture
    
    """
    def __init__(self):
        super(CnnModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 50, (5, 4), bias=False)
        self.batch1 = nn.BatchNorm2d(50, track_running_stats=False)
        self.maxpool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(50, 30, 3, bias=False)
        self.conv3 = nn.Conv1d(30, 30, 3, bias=False)
        self.flatt = nn.Flatten()
        self.dense1 = nn.Linear(1980, 500)
        self.dropout1 = nn.Dropout(p=0.3)
        self.dense2 = nn.Linear(500, 200)
        self.dense_output = nn.Linear(200, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.batch1(x)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2])
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.flatt(x)
        x = self.dense1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        output = self.dense_output(x)
        return output


# In[6]:


class Kbar(object):
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
            # if torch tensor, convert it to numpy
            if str(type(v)) == "<class 'torch.Tensor'>":
                v = v.detach().cpu().numpy()

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


# In[7]:


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='CNNModel_checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
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


# In[8]:


def load_data(path,batch,split_ratio = 0.2):
    print("*** Chargements des données ***")
    seqs_trainset_path = path + "seqs_trainset.npy"
    signals_trainset_path = path +"signals_trainset.npy"
    genenames_trainset_path = path + "genenames_trainset.npy"

    seqs_testset_path = path + "seqs_testset.npy"
    signals_testset_path = path + "signals_testset.npy"
    genenames_testset_path = path + "genenames_testset.npy"


    train_data = SeqDataset(seq_path=seqs_trainset_path, genenames_path = genenames_trainset_path, 
                            signals_path = signals_trainset_path )
    test_data = SeqDataset(seq_path = seqs_testset_path, genenames_path=genenames_testset_path,
                            signals_path = signals_testset_path)
    
    num_train = len(train_data)
    print("full train data :",num_train)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split_start = int(np.floor(split_ratio * num_train))
    train_idx, val_idx = indices[split_start:], indices[:split_start]
    print("Train data",len(train_idx))
    print("Valid data",len(val_idx))


    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_loader = DataLoader(dataset=train_data, shuffle=False, batch_size = batch, sampler=train_sampler)
    valid_loader = DataLoader(dataset = train_data, shuffle=False,batch_size=batch, sampler= val_sampler)
    test_loader = DataLoader(dataset=test_data, shuffle=True, batch_size = batch)
    
    print("Train data :",len(train_loader)*batch)
    print("Test data :",len(test_loader)*batch)
    print("Valid data :",len(valid_loader)*batch)


    return train_loader, valid_loader, test_loader


# In[9]:


def training_pytorch_model(custom_model, train_loader, valid_loader, batch, epoch, criterion, optimizer, device, patience=5, keep_track=True):
  
    print("*** Entrainement du modèle ***")

    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    # to track outputs
    early_id = 0

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    
    print("Nombre batchs training_set : ",len(train_loader))
    print("Nombre batchs valid_set : ",len(valid_loader))
    print("Taille batchs : ", batch)
    print("Nombre epochs : ", epoch)

    for e in range(epoch):

        running_loss = 0.000

        print(" ")
        print('Epoch: %d/%d' % (e + 1, epoch))
        
        kbar = Kbar(target=(len(train_loader) + len(valid_loader))*batch, width=12)
        
        custom_model.train()

        for i_batch, item in enumerate(train_loader):
            seq = item['seq'].to(device)
            signal = item['signal'].to(device)
            
            # Forward pass
            outputs = custom_model(seq)
            
            signal = signal.long()
            loss = criterion(outputs, signal)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())  
            train_loss = loss.item()
            kbar.update(i_batch * batch, values=[("train_loss", train_loss)])

        custom_model.eval()  # Sets the model in training mode, which changes the behaviour of dropout layers...
        with torch.no_grad():
            for i_val, item_val in enumerate(valid_loader):
                seq_valid = item_val['seq'].to(device)
                signal_valid = item_val['signal'].to(device)

                outputs_valid = custom_model(seq_valid)
                signal_valid = signal_valid.long() 
                loss_v = criterion(outputs_valid, signal_valid)

                valid_losses.append(loss_v.item())
                valid_loss = loss_v.item()
                kbar.update((i_val + i_batch) * batch, values=[("valid_loss", valid_loss)])
                
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
                
        msg = (f'train_loss: {train_loss:.5f} ' + f'valid_loss: {valid_loss:.5f}')
        print("")
        
        print(msg)
         
        train_losses = []
        valid_losses = []
        
        print("")
        
        early_stopping(valid_loss, custom_model)

        if early_stopping.early_stop:
            early_id = e - patience +1
            print("Early n ", early_id, "stopping")
            break

    custom_model.load_state_dict(torch.load('CNNModel_checkpoint.pt'))
    #torch.save(custom_model.state_dict(), "model_net.pth")

    if keep_track is True:
        return avg_train_losses, avg_valid_losses, early_id
    else :
        return 0


# In[10]:


def test_pytorch_model(custom_model, test_loader, device, batch, verbose=True):
    print("\n*** Test du modèle ***")
    classes = ('signal mauvais', 'signal correct')

    class_correct = list(0. for i in range(2))
    class_total = list(0. for i in range(2))
    
    custom_model.load_state_dict(torch.load('CNNModel_checkpoint.pt'))

    preds = []
    true = []
    correct = 0
    total = 0
    signals1 = 0
    signals0 = 0
    signals_list = []
    outputs_list = []
    custom_model.eval()  # prep model for evaluation
    with torch.no_grad():
        for i_test, item in enumerate(test_loader):
            seq = item['seq'].to(device)
            signals = item['signal'].to(device)
                    
            #signal = signal.view(-1, 1)

            # forward pass: compute predicted outputs by passing inputs to the model
            output = custom_model(seq)
            _, predicted = torch.max(output.data, 1)
            total += signals.size(0)
            
            correct += (predicted == signals).sum().item()
            signals1 += (signals == 1).sum().item()
            signals0 += (signals == 0).sum().item()
            
            c = (predicted == signals).squeeze()
            for i in range(signals.size(0)):
                signal = signals[i]
                class_correct[signal] += c[i].item()
                class_total[signal] += 1
            
            for signal in signals:
                signals_list.append(signal)
            for output in predicted :
                outputs_list.append(output)           
            

        print("--- Résultats du set de test ---")
        accu = 100 * correct / total
        print('Accuracy of the network ',accu, "%")
        
        print("Total elements classe 1 : ", 100 * (signals1/(signals1+signals0)), " %")
        print("Total elements classe 0 : ",  100 * (signals0/(signals1+signals0)), " %")

        for i in range(2):
            print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
            
        sig1percent = 100 * class_correct[1] / class_total[1]
        sig0percent = 100 * class_correct[0] / class_total[0]
        
        target_names = ['signal 0', 'signal 1']
        print(type(signals_list[0]))
        print(type(outputs_list[0]))

        report = classification_report(signals_list, outputs_list, target_names=target_names,output_dict=True)
               
        accuracy = report['accuracy']
        f1score_sig0 =  report['signal 0']['f1-score'] 
        f1score_sig1 =  report['signal 1']['f1-score'] 
        macro_precision = report['macro avg']['precision']  
        macro_recall = report['macro avg']['recall'] 
        macro_f1score = report['macro avg']['f1-score'] 
        
        print("\n --- Résultats du set de test ---\n")
        print("Accuracy (classification_report) : ",accuracy)
        print("\n Signal0 f1-score (classification_report) : ",f1score_sig0)
        print("Signal1 f1-score (classification_report) : ",f1score_sig1)
        print("\n macro AVG precision (classification_report) : ",macro_precision )
        print("macro AVG recall (classification_report) : ", macro_recall)
        print("macro AVG f1_score (classification_report) : ", macro_f1score )
        
        sig0tot = report['signal 0']['support'] 
        sig1tot = report['signal 1']['support'] 
        sigtot = report['macro avg']['support']
        
        print("\n Elements classe 0 in test_set : ",sig0tot)
        print("Elements classe 1 in test_set : ",sig1tot)
        print("Elements in test_set : ", sigtot)
        
        auc = roc_auc_score(signals_list, outputs_list)
        # summarize scores
        print('CNN Model: ROC AUC=%.3f' % (auc))
        
        fpr, tpr, thresholds = roc_curve(signals_list, outputs_list)
        plt.plot(fpr, tpr, linestyle='-', label='CNN ROC')
        
        # axis labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # show the legend
        plt.legend()
        # show the plot
        plt.show()
        #roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
        
      

    return accuracy, f1score_sig0, f1score_sig1


# In[12]:


def showLossCurve(avg_train_losses, avg_valid_losses,early_id,idSave,batch_size,epoch,accuracy, f1score_sig0, f1score_sig1):
    print("\n*** Affichage de la courbe de cout ***")

    imgname = 'img'+ str(idSave) + '_e'+ str(epoch)+'_b'+ str(batch_size)

    
    fig, axs = plt.subplots(figsize=(10, 10))

    plt.plot(range(1, len(avg_train_losses) + 1), avg_train_losses, label='Training Loss')
    plt.plot(range(1, len(avg_valid_losses) + 1), avg_valid_losses, label='Validation Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    axs.set_title(imgname)
    
    plt.ylim(0,max(avg_valid_losses) + 0.01 * max(avg_valid_losses))  # consistent scale
    if early_id != 0 :
        plt.axvline(early_id, linestyle='--', color='r', label='Early Stopping Checkpoint')
    plt.xlim(0, len(avg_train_losses) + 1)  # consistent scale
    plt.grid(True)
    accuracy = round(accuracy*100)
    f1score_sig0 = round(f1score_sig0*100)
    f1score_sig1 = round(f1score_sig1*100)
    text = "CNNsplit " + idSave + "\nAccuracy of the network : " + str(accuracy) + "\n (sig0: " + str(f1score_sig0) + " / sig1: " + str(f1score_sig1) + " )" 
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    axs.text(0.05, 0.95, text, transform=axs.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

    plt.legend()
    plt.tight_layout()
    plt.show()
    imgname = 'img'+ str(idSave) + '_e'+ str(epoch)+'_b'+ str(batch_size)
    path = "./results/" + imgname
    fig.savefig(path, bbox_inches='tight')


# In[13]:


#paramètres du modèle
path = "./datasetsplit/"
#path = "./dataset/"
batch_size = 200
epoch = 100
idSave = "originalmodel"
net = CnnModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[14]:


train_loader, valid_loader, test_loader = load_data(path,batch_size)
avg_train_losses, avg_valid_losses, early_id = training_pytorch_model(net, train_loader, valid_loader, batch_size, epoch, criterion, optimizer, device)
accuracy, f1score_sig0, f1score_sig1 = test_pytorch_model(net, test_loader, device, batch_size)
showLossCurve(avg_train_losses, avg_valid_losses, early_id,idSave,batch_size,epoch,accuracy, f1score_sig0, f1score_sig1 )
#print('AVG TRAIN LOSSES : ', avg_train_losses)
#print('AVG VALID LOSSES : ', avg_valid_losses)
