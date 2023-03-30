import torch
import numpy as np
from sklearn.metrics import roc_auc_score

def fit(train_loader, val_loader, model, loss_siamese_fn, loss_binary_fn, lr, n_epochs, device, log_interval, metrics=[],
        start_epoch=0):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    BEST_AUC = 0

    for epoch in range(start_epoch, n_epochs):
        '''
        if epoch > 600:
            lr_e = lr/10
        else:
            lr_e = lr
        '''
        lr_e = lr
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_e)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)

        # Train stage
        train_loss, acc, auc = train_epoch(train_loader, model, loss_siamese_fn, loss_binary_fn, optimizer, device, log_interval)
        scheduler.step()
        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}, Accuracy: {:.4f}, AUC: {:.4f}'.format(epoch + 1, n_epochs, train_loss, acc, auc)


        val_loss, auc = test_epoch(val_loader, model, loss_siamese_fn, loss_binary_fn, device)
        val_loss /= len(val_loader)

        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}, AUC: {:.4f}'.format(epoch + 1, n_epochs,
                                                                                 val_loss, auc)
        if epoch%5 == 0:
            print(message)

        if auc > BEST_AUC:
            BEST_AUC = auc
            print('NEW BEST AUC : {:.4f}!!!!'.format(BEST_AUC))
    print('Finished, Best AUC is {:.4f}'.format(BEST_AUC))


def train_epoch(train_loader, model, loss_siamese_fn, loss_binary_fn, optimizer, device, log_interval):

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
        outputs = outputs.squeeze(-1)
        outputs_total.extend(outputs)

        predicted = outputs.round()
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        loss_inputs = embeddings
        if target is not None:
            loss_inputs.append(target)

        _, loss_siamese = loss_siamese_fn(*loss_inputs)

        loss_binary = loss_binary_fn(outputs, target.float())


        loss_siamese.backward(retain_graph=True)
        loss_binary.backward()
        optimizer.step()
        #loss = loss_binary# + loss_siamese
        #losses.append(loss.item())
        #total_loss += loss.item()
        #loss.backward()
        #optimizer.step()


        '''
        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))

            print(message)
            losses = []
        '''

    total_loss /= (batch_idx + 1)
    auc = roc_auc_score(np.array(targets_total).astype(int), outputs_total)
    acc = 100. * correct / total
    return total_loss, acc, auc


def test_epoch(val_loader, model, loss_siamese_fn, loss_binary_fn, device):
    with torch.no_grad():
        model.eval()
        val_loss = 0
        outputs_total = []
        targets_total = []
        for batch_idx, (data, target) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            for i in range(len(data)):
                data[i] = data[i].to(device)
            target = target.to(device)
            targets_total.extend(target)
            embeddings, outputs = model(*data)
            outputs = outputs.squeeze(-1)
            outputs_total.extend(outputs)

            loss_inputs = embeddings
            if target is not None:
                loss_inputs.append(target)

            _, loss_siamese = loss_siamese_fn(*loss_inputs)

            loss_binary = loss_binary_fn(outputs, target.float())

            loss = loss_binary# + 0.01*loss_siamese
            val_loss += loss.item()

        auc = roc_auc_score(np.array(targets_total).astype(int), outputs_total)

    return val_loss, auc