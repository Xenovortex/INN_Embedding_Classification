import torch
import torch.nn as nn
from architecture import model
import numpy as np


class GenerativeClassifier(nn.Module):
    """
    taken from Lynton with slight modifications

    """
    def __init__(self, init_latent_scale=4, dims=(2, 2), n_classes=1000, lr=5e-4, weight_init=1.0, use_vgg=True):
        super().__init__()

        self.dims = dims
        self.n_classes = n_classes
        self.ndim_tot = int(np.prod(dims))
        
        if use_vgg:
            self.inn = model.inn_model(self.ndim_tot)
        else:
            self.inn = model.constuct_inn(self.dims)
        init_scale = init_latent_scale / np.sqrt(2 * self.ndim_tot // n_classes)
        self.mu = nn.Parameter(torch.zeros(1, n_classes, self.ndim_tot))
        for k in range(self.ndim_tot // n_classes):
            self.mu.data[0, :, n_classes * k : n_classes * (k+1)] = init_scale * torch.eye(n_classes)

        self.trainable_params = list(self.inn.parameters())
        self.trainable_params = list(filter(lambda p: p.requires_grad, self.trainable_params))

        self.lr = lr
        self.mu_lr = 20 * lr
        self.train_mu  = True
        self.train_inn = True

        for p in self.trainable_params:
            p.data *= weight_init

        self.trainable_params += [self.mu]

        optimizer_params = [ {'params':list(filter(lambda p: p.requires_grad, self.inn.parameters()))},]

        if self.train_mu:
            optimizer_params.append({'params': [self.mu],
                                     'lr': 0.1,
                                     'weight_decay': 0.,
                                     'betas': [0, 0.9]})

        self.optimizer = torch.optim.Adam(optimizer_params, self.lr, betas=[0.9, 0.99], weight_decay=1e-4)

    def mu_pairwise_dist(self):

        mu_i_mu_j = self.mu.squeeze().mm(self.mu.squeeze().t())
        mu_i_mu_i = torch.sum(self.mu.squeeze()**2, 1, keepdim=True).expand(self.n_classes, self.n_classes)

        dist =  mu_i_mu_i + mu_i_mu_i.t() - 2 * mu_i_mu_j
        return torch.masked_select(dist, (1 - torch.eye(self.n_classes).cuda()).byte()).clamp(min=0.)

    def forward(self, x, y=None, loss_mean=True):
        z = self.inn(x)
        jac = self.inn.log_jacobian(run_forward=False)

        zz = torch.sum((z.view(-1, 1, self.ndim_tot) - self.mu)**2, dim=2)

        losses = {'nll_joint_tr': (- torch.logsumexp(- 0.5 * zz - np.log(self.n_classes), dim=1) - jac ) / self.ndim_tot,
                  'logits_tr':    - 0.5 * zz}

        if y is not None:
            losses['nll_class_tr'] = (0.5 * torch.sum(zz * y, dim=1) - jac) / self.ndim_tot
            #losses['cat_ce_tr'] = - n_classes * torch.sum(torch.log_softmax(- 0.5 * zz, dim=1) * y, dim=1)
            #print(losses['nll_class_tr'].shape, losses['nll_joint_tr'].shape)
            losses['cat_ce_tr'] = (losses['nll_class_tr'] - losses['nll_joint_tr']) * self.ndim_tot

        if loss_mean:
            for k,v in losses.items():
                losses[k] = torch.mean(v)

        return losses

    def validate(self, x, y):
        is_train = self.inn.training
        self.inn.eval()
        with torch.no_grad():
            losses = self.forward(x, y, loss_mean=False)
            nll_joint, nll_class, cat_ce, logits = (losses['nll_joint_tr'].mean(),
                                                    losses['nll_class_tr'].mean(),
                                                    losses['cat_ce_tr'].mean(),
                                                    losses['logits_tr'])
            acc = torch.mean((torch.max(y, dim=1)[1] == torch.max(logits, dim=1)[1]).float())
            mu_dist = torch.mean(torch.sqrt(self.mu_pairwise_dist()))

        if is_train:
            self.inn.train()

        return {'nll_joint_test': nll_joint,
                'nll_class_test': nll_class,
                'logits_test':    logits,
                'cat_ce_test':    cat_ce,
                'acc_test':       acc,
                'delta_mu_test':  mu_dist}

    def sample(self, y, temperature=1.):
        z = temperature * torch.randn(y.shape[0], self.ndim_tot).cuda()
        mu = torch.sum(y.view(-1, self.n_classes, 1) * self.mu, dim=1)
        z = z + mu
        return self.inn(z, rev=True)

    def save(self, fname):
        torch.save({'inn':self.inn.state_dict(),
                    'mu':self.mu,
                    'opt':self.optimizer.state_dict()}, fname)

    def load(self, fname):
        data = torch.load(fname)
        self.inn.load_state_dict(data['inn'])
        self.mu.data.copy_(data['mu'].data)
        try:
            self.optimizer.load_state_dict(data['opt'])
        except:
            print('loading the optimizer went wrong, skipping')