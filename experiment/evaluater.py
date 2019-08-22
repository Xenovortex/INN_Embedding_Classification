# functions taken from Lynton

import glob

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
#from sklearn.manifold import TSNE
from matplotlib.backends.backend_pdf import PdfPages

#import model
import data


def show_samples(model, y, T=0.75):
    with torch.no_grad():
        samples = model.sample(y, T)
        samples = data.de_augment(samples).cpu().numpy()

    w = y.shape[1]
    h = int(np.ceil(y.shape[0] / w))

    plt.figure()
    for k in range(y.shape[0]):
        plt.subplot(h, w, k+1)
        plt.imshow(samples[k], cmap='gray')
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()


def show_latent_space(model, test_set=False):
    ''' the option `test_set` controls, whether the test set, or the validation set is used.'''

    clusters = model.mu.data.cpu().numpy().squeeze()
    pca = PCA(n_components=2)
    pca.fit(clusters)

    mu_red = pca.transform(clusters)
    z_red = []
    true_label = []

    data_generator = (data.test_loader if test_set else [(data.val_x, torch.argmax(data.val_y, dim=1))])

    with torch.no_grad():
        for x, y in data_generator:
            true_label.append(y.cpu().numpy())
            x, y = x.cuda(), data.onehot(y.cuda())
            z = model.inn(x).cpu().numpy()
            z_red.append(pca.transform(z))

    z_red = np.concatenate(z_red, axis=0)
    true_label= np.concatenate(true_label, axis=0)

    plt.figure()
    plt.scatter(mu_red[:,0], mu_red[:,1], c=np.arange(10), cmap='tab10', s=250, alpha=0.5)
    plt.scatter(z_red[:,0], z_red[:,1], c=true_label, cmap='tab10', s=1)
    plt.tight_layout()


def outlier_histogram(model, test_set=False):
    ''' the option `test_set` controls, whether the test set, or the validation set is used.'''

    data_generator = (data.test_loader if test_set else [(data.val_x, torch.argmax(data.val_y, dim=1))])

    score, correct_pred = [], []

    from torchvision.datasets import FashionMNIST
    from torch.utils.data import DataLoader
    fashion_generator  = DataLoader(FashionMNIST('./fashion_mnist', download=True, train=False, transform=data.transform),
                                    batch_size=data.batch_size, num_workers=8)

    # continue using dropout for WAIC
    model.train()

    def waic(x):
        waic_samples = 1
        ll_joint = []
        for i in range(waic_samples):
            losses = model(x, y=None, loss_mean=False)
            ll_joint.append(losses['nll_joint_tr'].cpu().numpy())

        ll_joint = np.stack(ll_joint, axis=1)
        return np.mean(ll_joint, axis=1) + np.var(ll_joint, axis=1)

    with torch.no_grad():
        for x, y in data_generator:
            x, y = x.cuda(), data.onehot(y.cuda())
            score.append(waic(x))
            losses = model(x, y, loss_mean=False)
            correct_pred.append((torch.argmax(y, dim=1)
                                 == torch.argmax(losses['logits_tr'], dim=1)).cpu().numpy())

        score_fashion = []
        for x, y in fashion_generator:
            x = x.cuda()
            x = data.augment(x)
            score_fashion.append(waic(x))

        score_adv = []
        score_adv_ref = []

        adv_images =  np.stack([np.load(f) for f in glob.glob('./adv_examples/adv_*.npy')], axis=0)
        ref_images = np.stack([np.load(f) for f in glob.glob('./adv_examples/img_*.npy')], axis=0)

        adv_images = data.augment(torch.Tensor(adv_images).cuda())
        ref_images = data.augment(torch.Tensor(ref_images).cuda())

        score_adv = waic(adv_images)
        score_ref = waic(ref_images)

    model.eval()

    score = np.concatenate(score, axis=0)
    correct_pred = np.concatenate(correct_pred, axis=0)
    score_fashion = np.concatenate(score_fashion, axis=0)

    #val_range = [np.quantile(np.concatenate((score, score_fashion, score_adv)),  0.01),
                 #np.quantile(np.concatenate((score, score_fashion, score_adv)),  0.6)]
    val_range = [-8, 0]
    val_range[0] -= 0.03 * (val_range[1] - val_range[0])
    val_range[1] += 0.03 * (val_range[1] - val_range[0])

    bins = 40

    plt.figure()
    plt.hist(score[correct_pred == 1], bins=bins,  range=val_range, histtype='step', label='correct', density=True, color='green')
    plt.hist(score[correct_pred == 0], bins=3*bins,range=val_range, histtype='step', label='not correct', density=True, color='red')
    plt.hist(score_fashion,            bins=bins,  range=val_range, histtype='step', label='$\mathcal{Fashion}$', density=True, color='magenta')

    plt.hist(score_adv               , bins=bins,  range=val_range, histtype='step', label='Adv attacks', density=True, color='blue')
    plt.hist(score_ref               , bins=bins,  range=val_range, histtype='step', label='Non-attacked images', density=True, color='gray')

    plt.legend()


def calibration_curve(model):

    pred = []
    gt = []
    with torch.no_grad():
        for x, y in data.test_loader:
            x, y = x.cuda(), data.onehot(y.cuda())
            logits = model(x, y, loss_mean=False)['logits_tr']

            #log_pred = logits - torch.logsumexp(logits + np.log(1/10.), dim=1, keepdim=True)
            #exp_pred = torch.exp(log_pred)
            #print(torch.mean(torch.max(exp_pred, dim=1)[0]))

            pred.append(torch.softmax(logits, dim=1).cpu().numpy())
            #pred.append(torch.exp(log_pred).cpu().numpy())
            gt.append(y.cpu().numpy())

    pred = np.concatenate(pred, axis=0).flatten()
    gt   = np.concatenate(gt,   axis=0).astype(np.bool).flatten()

    mask = (pred > 1e-6)
    mask = mask * (pred < (1 - 1e-6))

    pred, gt = pred[mask], gt[mask]

    n_bins = np.sum(mask) / 50.
    pred_bins = np.quantile(pred, np.linspace(0., 1., n_bins))
    print(np.sum(mask))

    correct = pred[gt]
    wrong = pred[np.logical_not(gt)]

    hist_correct, _ = np.histogram(correct, bins=pred_bins)
    hist_wrong, _   = np.histogram(wrong,   bins=pred_bins)

    q = hist_correct / (hist_wrong + hist_correct)
    p = 0.5 * (pred_bins[1:] + pred_bins[:-1])

    poisson_err = q * np.sqrt( 1 / hist_correct + 1 / (hist_wrong + hist_correct))

    plt.figure(figsize=(10, 10))
    plt.errorbar(p, q, yerr=poisson_err, capsize=4, fmt='-o')
    plt.fill_between(p, q - poisson_err, q + poisson_err, alpha=0.25)
    plt.plot([0,1], [0,1], color='black')


def val_plots(fname, model):
    n_classes = data.n_classes
    n_samples = 4

    y_digits = torch.zeros(n_classes * n_samples, n_classes).cuda()
    for i in range(n_classes):
        y_digits[n_samples * i : n_samples * (i+1), i] = 1.

    show_samples(model, y_digits)
    show_latent_space(model)

    with PdfPages(fname) as pp:
        figs = [plt.figure(n) for n in plt.get_fignums()]
        for fig in figs:
            fig.savefig(pp, format='pdf')

    plt.close('all')