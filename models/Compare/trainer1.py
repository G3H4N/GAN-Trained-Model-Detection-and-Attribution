import torch
import numpy as np
from sklearn.metrics import roc_auc_score

def fit(train_loader, val_loader, model, loss_siamese_fn, loss_classify_fn, lr, n_epochs, device, log_interval, metrics=[],
        start_epoch=0):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    BEST_ACC = 0

    for epoch in range(start_epoch, n_epochs):
        '''
        if epoch > 350:
            lr_e = lr/10
        else:
            lr_e = lr
        '''
        lr_e = lr
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)#Adam(model.parameters(), lr=lr_e)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)

        # Train stage
        train_loss, acc = train_epoch(epoch, train_loader, model, loss_siamese_fn, loss_classify_fn, optimizer, device, log_interval)
        scheduler.step()
        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(epoch + 1, n_epochs, train_loss, acc)


        val_loss, acc = test_epoch(val_loader, model, loss_siamese_fn, loss_classify_fn, device)
        val_loss /= len(val_loader)

        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}, ACC: {:.4f}'.format(epoch + 1, n_epochs,
                                                                                 val_loss, acc)
        if epoch%5 == 0:
            print(message)

        if acc > BEST_ACC:
            BEST_ACC = acc
            print('NEW BEST AUC : {:.4f}!!!!'.format(BEST_ACC))
    print('Finished, Best AUC is {:.4f}'.format(BEST_ACC))


def train_epoch(epoch, train_loader, model, loss_siamese_fn, loss_classify_fn, optimizer, device, log_interval):

    model.train()
    losses = []
    total_loss = 0
    outputs_total = []
    targets_total = []
    total = 0
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        for i in range(len(data)):
            data[i] = data[i].to(device)
        target = target.to(device)

        targets_total.extend(target)

        optimizer.zero_grad()
        embeddings, outputs = model(*data)
        #outputs = outputs.squeeze(-1)
        outputs_total.extend(outputs)

        _, predicted = outputs.max(1)
        predicted = predicted.to(device)
        correct += predicted.eq(target).sum().item()
        total += target.size(0)

        loss_inputs = embeddings
        if target is not None:
            loss_inputs.append(target)

        _, loss_siamese = loss_siamese_fn(*loss_inputs)

        loss_classify = loss_classify_fn(outputs, target)

        '''
        loss_siamese.backward(retain_graph=True)
        loss_classify.backward()
        optimizer.step()
        '''
        loss = loss_classify# + loss_siamese
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()


        '''
        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))

            print(message)
            losses = []
        '''

    total_loss /= (batch_idx + 1)
    #auc = roc_auc_score(np.array(targets_total).astype(int), outputs_total)
    acc = 100. * correct / total
    return total_loss, acc


def test_epoch(val_loader, model, loss_siamese_fn, loss_classify_fn, device):
    with torch.no_grad():
        model.eval()
        val_loss = 0
        outputs_total = []
        targets_total = []
        total = 0
        correct = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            for i in range(len(data)):
                data[i] = data[i].to(device)
            target = target.to(device)
            targets_total.extend(target)
            embeddings, outputs = model(*data)
            outputs = outputs.squeeze(-1)
            outputs_total.extend(outputs)

            _, predicted = outputs.max(1)
            predicted = predicted.to(device)
            correct += predicted.eq(target).sum().item()
            total += target.size(0)

            loss_inputs = embeddings
            if target is not None:
                loss_inputs.append(target)

            _, loss_siamese = loss_siamese_fn(*loss_inputs)

            loss_classify = loss_classify_fn(outputs, target)

            loss = loss_siamese+loss_classify# + 0.01*loss_siamese
            val_loss += loss.item()

        #auc = roc_auc_score(np.array(targets_total).astype(int), outputs_total)
        acc = 100. * correct / total

    return val_loss, acc