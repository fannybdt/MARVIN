import os
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR

from sklearn.metrics import f1_score, balanced_accuracy_score

import wandb

class Concat(nn.Module):
    def __init__(self, f_in, f_out):
        super().__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.projs = nn.ModuleList([nn.Linear(f, f_out) for f in f_in])

    def forward(self, xs):
        return torch.cat([proj(x) for proj, x in zip(self.projs, xs)], dim=-1)

class ResidualBlock(nn.Module):
    def __init__(self, f_in, f_out, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(f_in, f_out),
            nn.ReLU(),
            nn.Linear(f_out, f_out),
            nn.Dropout(dropout)
        )

        self.shortcut = nn.Identity() if f_in == f_out else nn.Linear(f_in, f_out)

    def forward(self, x):
        return F.relu(self.net(x) + self.shortcut(x))

class ResidualNetwork(nn.Module):
    def __init__(self, f_in, f_out, factor, dropout, num_blocks):
        super().__init__()
        layers = [ResidualBlock(f_in, factor*f_in, dropout)]
        for _ in range(num_blocks - 1):
            layers.append(ResidualBlock(factor*f_in, factor*f_in, dropout))
        layers.append(nn.Linear(factor*f_in, f_out))
        self.blocks = nn.ModuleList(layers)
        
    def forward(self, x):
        y = x
        for block in self.blocks:
            y = block(y)
        return y

class MARVIN(nn.Module):
    def __init__(self, M, K, D, factor, dropout, num_blocks):
        super().__init__()
        self.M = M
        self.K = K
        self.D = D
        
        self.factor = factor
        self.dropout = dropout
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # q(c, z|x) = q(c|x) q(z|c, x)
        self.q_c_x = ResidualNetwork(M, K, factor, dropout, num_blocks)

        self.q_z_cx = nn.Sequential(
            Concat([K, M], (K+M) // 2),
            # nn.Linear(K + M, K + M),
            nn.ReLU(),
            ResidualNetwork((K+M)//2 *2, 2 * D, factor, dropout, num_blocks)
        )
        self.p_c = nn.Parameter(torch.zeros(self.K))

        self.p_z_c = ResidualNetwork(K, 2 * D, factor, dropout, num_blocks)

        self.p_x_z = ResidualNetwork(D, 2 * M, factor, dropout, num_blocks)

        
    def loss_unsupervised(self, x):
        # Assume a minibatch x, _ = next(iter(loader))
        
        # ELBO = E_p(x) E_q(c, z|x) [log p(x, c, z) - log q(c, z|x)]
        # = E_p(x) E_q(c, z|x) [ log p(x | z) ] + E_p(x) [ KL(q(c, z | x) || p(c, z)) ]

        # iterate over the K clusters
        elbo = 0.

        p_c = F.softmax(self.p_c, dim=0)
        p_c = torch.clamp(p_c, min = 1e-6)
        q_c_x = F.softmax(self.q_c_x(x), dim=-1)
        q_c_x = torch.clamp(q_c_x, min=1e-6)
       
        for k in range(self.K):
            # c, z ~ q(c=k, z | x) Marginalization over clusters
            c = torch.tensor([k] * len(x)).to(self.device)
            codes = self.one_hot(c, self.K).to(self.device)
            q_z_cx = self.q_z_cx([codes, x])
            mu_z = q_z_cx[:, :self.D]
            logvar_z = q_z_cx[:, self.D:] 
            z = mu_z + logvar_z.exp() ** 0.5 * torch.randn_like(mu_z)

            # E_p(x) E_q(c, z|x) [ log p(x | z) ]
            x_recon = self.p_x_z(z)
            mu_x = x_recon[:, :self.M]
            logvar_x = x_recon[:, self.M:] 
            logvar_x = torch.clamp(logvar_x, min=-6, max=3)
            elbo +=  q_c_x[:, k] * (0.5* ((x - mu_x) ** 2 / logvar_x.exp() + logvar_x)).sum(dim=-1)

            # E_p(x) [ KL(q(c, z | x) || p(c, z)) ] max -KL = min KL
            p_z_c = self.p_z_c(codes)
            elbo +=  q_c_x[:, k] * (q_c_x[:, k].log() - p_c[k].log())
            elbo +=  q_c_x[:, k] * self.kl_gaussian(mu_z, logvar_z, p_z_c[:, :self.D], p_z_c[:, self.D:])

        return elbo.mean()

    def loss_supervised(self, x, c):
        # Assume a minibatch x, c = next(iter(loader)), cross entropy loss
        q_c_x = self.q_c_x(x)
        return F.cross_entropy(q_c_x, c)

    def one_hot(self, c, K):
        return torch.eye(K, device = c.device)[c]

    def kl_gaussian(self, mu_q, logvar_q, mu_p, logvar_p):
        # KL(q || p) between two Gaussian distributions
        return (0.5*(logvar_p - logvar_q) + 0.5 * (logvar_q.exp() + (mu_q - mu_p) ** 2) / logvar_p.exp() - 0.5).sum(dim=-1)
    
    
class MyDataset():
    def __init__(self, root, batch_size, M, ratio_supervision):
        self.root = root
        self.M = M
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ratio_supervision = ratio_supervision
    
    def loader(self):
        dataset = pd.read_csv(self.root)
        dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

        labels, uniques = pd.factorize(dataset['label'], sort=True)
        x = dataset.iloc[:, :self.M]
        n = int(0.8 * len(dataset))
        n2 = int(0.9 * len(dataset))

        # Split the dataset into training, validation, and test sets
        x_train_full = x.iloc[:n]
        c_train_full = pd.Series(labels[:n])
        x_validation = torch.tensor(x.iloc[n:n2].values, dtype=torch.float32)
        c_validation = torch.tensor(labels[n:n2], dtype=torch.long)
        x_test = torch.tensor(x.iloc[n2:].values, dtype=torch.float32)
        c_test = torch.tensor(labels[n2:], dtype=torch.long)

        keep_indices = []
        for label in np.unique(c_train_full):
            class_indices = c_train_full[c_train_full == label].index
            # minimum one sample per class
            n_keep = max(1, int(self.ratio_supervision * len(class_indices)))  
            selected = np.random.choice(class_indices, size=n_keep, replace=False)
            keep_indices.extend(selected)

        # c < 0 will not go through the supervised loss
        c_train_masked = pd.Series([-1] * len(c_train_full), index=c_train_full.index)
        c_train_masked.loc[keep_indices] = c_train_full.loc[keep_indices]

        x_train = torch.tensor(x_train_full.values, dtype=torch.float32)
        c_train_masked = torch.tensor(c_train_masked.values, dtype=torch.long)

        trainset = torch.utils.data.TensorDataset(x_train, c_train_masked)
        validationset = torch.utils.data.TensorDataset(x_validation, c_validation)
        testset = torch.utils.data.TensorDataset(x_test, c_test)

        trainloader = DataLoader(trainset, self.batch_size, shuffle=True, num_workers=4, drop_last=True)
        validationloader = DataLoader(validationset, self.batch_size, shuffle=True, num_workers=4, drop_last=True)
        testloader = DataLoader(testset, self.batch_size, shuffle=False)

        return trainloader, validationloader, testloader

class WarmupScheduler:
    def __init__(self, optimizer, warmup_steps, base_lr, max_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            lr = self.base_lr + (self.max_lr - self.base_lr) * (self.current_step / self.warmup_steps)
        else:
            lr = self.max_lr 

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        
    
class MARVINPipeline(nn.Module):
    def __init__(self, M, K, D, batch_size, eval_every, num_epochs, root, save_every, factor = 4, dropout = 0.2, lr = 1e-3, 
                 warmup_steps = 5000, base_lr = 1e-5, num_blocks = 2, ID = None, verbose = True, ratio_supervision = 1):
        super().__init__()

        self.M = M
        self.K = K
        self.D = D


        self.verbose = verbose
        self.batch_size = batch_size
        self.eval_every = eval_every
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.lr = lr
        self.root = root
        self.save_every = save_every
        self.dropout = dropout
        self.factor = factor
        self.num_blocks = num_blocks
        self.ratio_supervision = ratio_supervision

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.marvin = MARVIN(M, K, D, factor, dropout, num_blocks)
        self.marvin = self.marvin.to(self.device)
        
        self.ID = ID
        self.directory = "wandb_follow/"

        wandb.init(
            dir = self.directory,
            project = "TFE",
            config = {
                "model": f"MARVIN_{self.ID}",
                "batch_size": self.batch_size,
                "num_epochs": self.num_epochs,
                "learning_rate": self.lr,
                "eval_every": self.eval_every, 
                "factor": self.factor,
                "D" : self.D,
                "lr": self.lr
            }
        )

        self.optimizer =  torch.optim.AdamW(self.marvin.parameters(), lr=self.lr)
        self.warmup_scheduler = WarmupScheduler(self.optimizer, warmup_steps=self.warmup_steps, base_lr=self.base_lr, max_lr=self.lr)
        self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.5)
        self.dataset =  MyDataset(self.root, self.batch_size, self.M, self.ratio_supervision)
        self.trainloader, self.validationloader, self.testloader = self.dataset.loader()

    def train(self):
        print("\n\n= \t = \t = \t =")
        print(f" MARVIN {self.ID} begins training and has {sum(p.numel() for p in self.marvin.parameters())/1e6} M parameters", flush=True)
        print("= \t = \t = \t =")
        
        step_count = 0
        for epoch in range(self.num_epochs):
            self.marvin.train()
            for x, c in self.trainloader:
                c = c.to(self.device)
                x = x.to(self.device)
                
                self.optimizer.zero_grad()

                loss = self.marvin.loss_unsupervised(x)

                supervised = c >= 0
                x_sup = x[supervised]
                c_sup = c[supervised]
                if len(x_sup) > 0:
                    loss += self.marvin.loss_supervised(x_sup, c_sup)
                    
                loss.backward()
                #nn.utils.clip_grad_norm_(self.marvin.parameters(), max_norm = 10.0) #required for POISED dataset

                if step_count < self.warmup_steps:
                    self.warmup_scheduler.step()

                self.optimizer.step()
                step_count += 1
                if step_count % self.eval_every == 0:
                    for name, param in self.marvin.named_parameters():
                        if param.grad is not None:
                            # Log gradients for each parameter to W&B
                            wandb.log({f"grad_{name}": wandb.Histogram(param.grad.cpu().numpy())})
                    train_loss, validation_loss = self.eval()
                    accuracy, f1score, balanced_accuracy = self.metrics()
                    self.marvin.train()
                    wandb.log({"epoch":epoch+1,"step":step_count, "train_loss": train_loss, "validation_loss": validation_loss,
                    "accuracy": accuracy, "F1-score": f1score, "balanced accuracy": balanced_accuracy})
                    print(f"Epoch {epoch + 1}, Step {step_count}, validation loss = {validation_loss}, train loss = {train_loss}", flush = True)
                if step_count % self.save_every == 0:
                    self.save(f"{self.directory}/GMVAE_{self.ID}_{step_count}.pt")
            self.scheduler.step()
        test_loss, accuracy, balanced_accuracy, f1score = self.eval_on_testset()

        wandb.log({ "test_loss": test_loss,
                   "accuracy : test": accuracy, "F1-score : test": f1score, "balanced accuracy : test": balanced_accuracy})
        
                    
    def eval(self):
        with torch.no_grad():
            for k in {0, 1}:
                if k == 0:
                    loader = self.trainloader
                else:
                    loader = self.validationloader
                        
                self.marvin.eval()
                total_loss = 0
                i = 0
                for i, (x, c) in enumerate(loader):

                    x = x.to(self.device)
                    c = c.to(self.device)
                    loss = self.marvin.loss_unsupervised(x)
                    
                    supervised = c >= 0
                    x_sup = x[supervised]
                    c_sup = c[supervised]
                    if len(x_sup) > 0:
                        loss += self.marvin.loss_supervised(x_sup, c_sup)
                    total_loss += loss
                    if i == 19:
                        break
                    
                if k == 0:
                    train_loss = total_loss/(i + 1)
                else:
                    validation_loss = total_loss/(i + 1)

        return train_loss, validation_loss

    def metrics(self):
        with torch.no_grad():
            self.marvin.eval()
            accuracy = 0
            f1score = 0
            balanced_accuracy = 0
            i = 0
            loader = self.validationloader
            
            for i, (x_true, c_true) in enumerate(loader):
                x_true = x_true.to(self.device)
                c_true = c_true.to(self.device)
                q_c_x = F.softmax(self.marvin.q_c_x(x_true), dim=-1)
                c = torch.multinomial(q_c_x, 1, replacement = True).squeeze()
                
                accuracy += (c == c_true).sum().item() / c_true.size(0)
                f1score += f1_score(c_true.cpu().numpy(), c.cpu().numpy(), average='macro')
                balanced_accuracy += balanced_accuracy_score(c_true.cpu().numpy(), c.cpu().numpy())
                if i == 19:
                    break

            accuracy/=(i+1)
            f1score/=(i+1)
            balanced_accuracy/=(i+1)
        return accuracy, f1score, balanced_accuracy
    
    def eval_on_testset(self):
        
        loader = self.testloader
        
        test_loss = 0
        accuracy = 0
        f1score = 0
        balanced_accuracy = 0
        
        for x, c_true in loader:
            x = x.to(self.device)
            c_true = c_true.to(self.device)
            

            loss = self.marvin.loss_unsupervised(x)

            supervised = c_true >= 0
            x_sup = x[supervised]
            c_sup = c_true[supervised]

            if len(x_sup) > 0:
                loss += self.marvin.loss_supervised(x_sup, c_sup)
            test_loss += loss
            
            q_c_x = F.softmax(self.marvin.q_c_x(x), dim=-1)
            c = torch.multinomial(q_c_x, 1, replacement = True).squeeze()
            
            accuracy += (c == c_true).sum().item() / c_true.size(0)
            f1score += f1_score(c_true.cpu().numpy(), c.cpu().numpy(), average='macro')
            balanced_accuracy += balanced_accuracy_score(c_true.cpu().numpy(), c.cpu().numpy())
            
        test_loss /= len(loader)
        accuracy /= len(loader)
        f1score /= len(loader)
        balanced_accuracy /= len(loader)
        
        return test_loss, accuracy, balanced_accuracy, f1score
            
            
    def save(self, checkpoint):
        torch.save(self.marvin.state_dict(), checkpoint)
        print(f"Model saved to {checkpoint}")
        
    def load(self, checkpoint):
        self.marvin.load_state_dict(torch.load(checkpoint))
        self.marvin.to(self.device)
        print(f"Model loaded from {checkpoint}")




        














