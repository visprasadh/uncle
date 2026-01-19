"""
From https://github.com/vikram2000b/bad-teaching-unlearning / https://arxiv.org/abs/2205.08096
"""

import torch
import numpy as np
from torch.nn import functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def get_x_y_from_data_dict(data, device):
    x, y = data.values()
    if isinstance(x, list):
        x, y = x[0].to(device), y[0].to(device)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


class UnlearningMetricCalculator:
    def __init__(
        self,
        device,
        hnet=False,
    ):
        self.device = device
        self.hnet = hnet

    def get_output(self, tuple_model, task_id, data, lwsf=False):
        unique_task_id = task_id.unique()
        y_pred = torch.zeros(data.shape[0], 10).to(self.device)
        for t in unique_task_id:
            indices = torch.where(t == task_id)
            task_data = data[indices]
            if self.hnet:
                task_embedding = tuple_model[0][int(t.item())]
                hnet, mnet = tuple_model[1], tuple_model[2]
                w, b, nw, nb, dw, db = hnet(task_embedding)
                y_pred_t = mnet(task_data, w, b, nw, nb, dw, db)
                y_pred[indices] = y_pred_t
            else:
                y_pred_t = tuple_model(task_data, int(t.item()), lwsf=lwsf)
                y_pred[indices] = y_pred_t
        return y_pred

    def JSDiv(self, p, q):
        m = (p + q) / 2
        return 0.5 * F.kl_div(
            torch.log(p),
            m,
            reduction="batchmean",
        ) + 0.5 * F.kl_div(
            torch.log(q),
            m,
            reduction="batchmean",
        )

    def UnLearningScore(
        self,
        tmodel,
        forget_dl,
        device,
        forget_task=None,
        lwsf=False,
    ):
        model_preds = []
        gold_model_preds = []
        with torch.no_grad():
            for batch in forget_dl:
                x, _, _ = batch
                x = x.to(device)
                task_ids = torch.ones(x.shape[0]).to(device) * forget_task
                model_output = self.get_output(tmodel, task_ids, x, lwsf=lwsf)
                model_preds.append(F.softmax(model_output, dim=1).detach().cpu())
                # comaprision with uniform distribution
                gold_model_preds.append(
                    F.softmax(torch.zeros_like(model_output), dim=1).detach().cpu()
                )
        model_preds = torch.cat(model_preds, axis=0)
        gold_model_preds = torch.cat(gold_model_preds, axis=0)
        return 1 - self.JSDiv(model_preds, gold_model_preds)

    def entropy(self, p, dim=-1, keepdim=False):
        return -torch.where(
            p > 0,
            p * p.log(),
            p.new([0.0]),
        ).sum(dim=dim, keepdim=keepdim)

    def collect_prob(self, dataset, model, lwsf=False, forget_task=None):
        prob, targets = [], []
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=512, shuffle=False
        )
        with torch.no_grad():
            for batch in data_loader:
                batch = [tensor.to(self.device) for tensor in batch]
                data, labels, t = batch
                if forget_task is not None:
                    t = torch.ones_like(t) * forget_task
                output = self.get_output(model, t, data, lwsf=lwsf)
                if isinstance(output, tuple):
                    output = output[0]
                prob.append(F.softmax(output, dim=-1).data)
                targets.append(labels)
        return torch.cat(prob, axis=0), torch.cat(targets, axis=0)

    def get_membership_attack_data(
        self,
        forget_task,
        retain_set,
        forget_set,
        test_set,
        model,
        lwsf=False,
    ):
        retain_prob = self.collect_prob(retain_set, model, lwsf=lwsf)
        forget_prob = self.collect_prob(
            forget_set,
            model,
            lwsf=lwsf,
            forget_task=forget_task,
        )
        test_prob = self.collect_prob(test_set, model, lwsf=lwsf)

        X_r = (
            torch.cat([self.entropy(retain_prob), self.entropy(test_prob)])
            .cpu()
            .numpy()
            .reshape(-1, 1)
        )
        Y_r = np.concatenate([np.ones(len(retain_prob)), np.zeros(len(test_prob))])

        X_f = self.entropy(forget_prob).cpu().numpy().reshape(-1, 1)
        Y_f = np.concatenate([np.ones(len(forget_prob))])
        return X_f, Y_f, X_r, Y_r

    def get_membership_attack_prob(
        self,
        forget_task,
        retain_set,
        forget_set,
        test_set,
        model,
        lwsf=False,
    ):
        if len(retain_set) == 0:
            return -1
        X_f, _, X_r, Y_r = self.get_membership_attack_data(
            forget_task,
            retain_set,
            forget_set,
            test_set,
            model,
            lwsf=lwsf,
        )
        # clf = SVC(C=3,gamma='auto',kernel='rbf')
        clf = LogisticRegression(
            class_weight="balanced", solver="lbfgs", multi_class="multinomial"
        )
        clf.fit(X_r, Y_r)
        results = clf.predict(X_f)
        return results.mean()

    def get_the_metrics(
        self,
        model,
        forget_task,
        forget_train_loader,
        forget_test_loader,
        retain_set,
        test_set,
        lwsf=False,
    ):
        uniform = self.UnLearningScore(
            model, forget_test_loader, self.device, forget_task, lwsf
        )
        mia_dict = self.SVC_MIA(
            shadow_train=retain_set,
            target_train=forget_train_loader.dataset,
            target_test=forget_test_loader.dataset,
            shadow_test=test_set,
            model=model,
            forget_task=forget_task,
            lwsf=lwsf,
        )
        return uniform, mia_dict

    def SVC_fit_predict(
        self,
        shadow_train,
        shadow_test,
        target_train,
        target_test,
    ):
        n_shadow_train = shadow_train.shape[0]
        n_shadow_test = shadow_test.shape[0]
        n_target_train = target_train.shape[0]
        n_target_test = target_test.shape[0]

        X_shadow = (
            torch.cat([shadow_train, shadow_test])
            .cpu()
            .numpy()
            .reshape(n_shadow_train + n_shadow_test, -1)
        )
        Y_shadow = np.concatenate([np.ones(n_shadow_train), np.zeros(n_shadow_test)])

        clf = SVC(C=3, gamma="auto", kernel="rbf")
        clf.fit(X_shadow, Y_shadow)

        accs = []

        if n_target_train > 0:
            X_target_train = target_train.cpu().numpy().reshape(n_target_train, -1)
            acc_train = clf.predict(X_target_train).mean()
            accs.append(acc_train)

        if n_target_test > 0:
            X_target_test = target_test.cpu().numpy().reshape(n_target_test, -1)
            acc_test = 1 - clf.predict(X_target_test).mean()
            accs.append(acc_test)

        return np.mean(accs)

    def SVC_MIA(
        self,
        shadow_train,
        target_train,
        target_test,
        shadow_test,
        model,
        forget_task,
        lwsf=False,
    ):
        shadow_train_prob, shadow_train_labels = self.collect_prob(
            shadow_train, model, lwsf=lwsf
        )
        shadow_test_prob, shadow_test_labels = self.collect_prob(
            shadow_test, model, lwsf=lwsf
        )

        target_train_prob, target_train_labels = self.collect_prob(
            target_train, model, lwsf=lwsf, forget_task=forget_task
        )
        target_test_prob, target_test_labels = self.collect_prob(
            target_test, model, lwsf=lwsf, forget_task=forget_task
        )

        shadow_train_corr = (
            torch.argmax(shadow_train_prob, axis=1) == shadow_train_labels
        ).int()
        shadow_test_corr = (
            torch.argmax(shadow_test_prob, axis=1) == shadow_test_labels
        ).int()
        target_train_corr = (
            torch.argmax(target_train_prob, axis=1) == target_train_labels
        ).int()
        target_test_corr = (
            torch.argmax(target_test_prob, axis=1) == target_test_labels
        ).int()

        shadow_train_conf = torch.gather(
            shadow_train_prob, 1, shadow_train_labels[:, None]
        )
        shadow_test_conf = torch.gather(
            shadow_test_prob, 1, shadow_test_labels[:, None]
        )
        target_train_conf = torch.gather(
            target_train_prob, 1, target_train_labels[:, None]
        )
        target_test_conf = torch.gather(
            target_test_prob, 1, target_test_labels[:, None]
        )

        shadow_train_entr = self.entropy(shadow_train_prob)
        shadow_test_entr = self.entropy(shadow_test_prob)

        target_train_entr = self.entropy(target_train_prob)
        target_test_entr = self.entropy(target_test_prob)

        acc_corr = self.SVC_fit_predict(
            shadow_train_corr,
            shadow_test_corr,
            target_train_corr,
            target_test_corr,
        )
        acc_conf = self.SVC_fit_predict(
            shadow_train_conf,
            shadow_test_conf,
            target_train_conf,
            target_test_conf,
        )
        acc_entr = self.SVC_fit_predict(
            shadow_train_entr,
            shadow_test_entr,
            target_train_entr,
            target_test_entr,
        )
        acc_prob = self.SVC_fit_predict(
            shadow_train_prob,
            shadow_test_prob,
            target_train_prob,
            target_test_prob,
        )
        m = {
            "correctness": acc_corr,
            "confidence": acc_conf,
            "entropy": acc_entr,
            "prob": acc_prob,
        }

        return m