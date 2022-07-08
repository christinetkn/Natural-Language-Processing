import math
import sys
import torch
import numpy as np

def progress(loss, epoch, batch, batch_size, dataset_size):
    """
    Print the progress of the training for each epoch
    """
    batches = math.ceil(float(dataset_size) / batch_size)
    count = batch * batch_size
    bar_len = 40
    filled_len = int(round(bar_len * count / float(dataset_size)))

    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    status = 'Epoch {}, Loss: {:.4f}'.format(epoch, loss)
    _progress_str = "\r \r [{}] ...{}".format(bar, status)
    sys.stdout.write(_progress_str)
    sys.stdout.flush()

    if batch == batches:
        print()


def train_dataset(_epoch, dataloader, model, loss_function, optimizer):
    # IMPORTANT: switch to train mode
    # enable regularization layers, such as Dropout
    model.train()
    running_loss = 0.0

    # obtain the model's device ID
    device = next(model.parameters()).device
    print("DEVISE IS:*********************** ", device) 

    for index, batch in enumerate(dataloader, 1):
        # get the inputs (batch)
        
        inputs, labels, lengths = batch
        inputs, labels, lengths = inputs.to(device).long(), labels.to(device), lengths.to(device)
        
        # move the batch tensors to the right device
        # EX9

        # Step 1 - zero the gradients
        # Remember that PyTorch accumulates gradients.
        # We need to clear them out before each batch!
        ...  # EX9

        # Step 2 - forward pass: y' = model(x)
        # EX9

        # Step 3 - compute loss: L = loss_function(y, y')
        # EX9

        # Step 4 - backward pass: compute gradient wrt model parameters
        ...  # EX9

        # Step 5 - update weights
        # EX9
        model.to(device) # EX9
        optimizer.zero_grad() # EX9
        outputs = model(inputs, lengths)
        
        label = torch.FloatTensor([[1,0] if i == 0 else [0,1] for i in labels])
        #label = torch.Tensor(np.array(labels, dtype = np.long)).long()
        #labels = labels.long()
        #labels = labels.float()
       

        loss = loss_function(outputs.to(device), label.to(device)) # EX9
        loss.backward()
        optimizer.step()
        

        

        
        running_loss += loss.data.item()
        

        # print statistics
        progress(loss=loss.data.item(),
                 epoch=_epoch,
                 batch=index,
                 batch_size=dataloader.batch_size,
                 dataset_size=len(dataloader.dataset))

    return running_loss / index


def eval_dataset(dataloader, model, loss_function):
    # IMPORTANT: switch to eval mode
    # disable regularization layers, such as Dropout
    model.eval()
    running_loss = 0.0

    y_pred = []  # the predicted labels
    y = []  # the gold labels

    # obtain the model's device ID
    device = next(model.parameters()).device

    # IMPORTANT: in evaluation mode, we don't want to keep the gradients
    # so we do everything under torch.no_grad()
    with torch.no_grad():
        for index, batch in enumerate(dataloader, 1):
            # get the inputs (batch)
            inputs, labels, lengths = batch
            inputs, labels, lengths = inputs.to(device).long(), labels.to(device), lengths.to(device)
            # Step 1 - move the batch tensors to the right device
            
            if torch.cuda.is_available():
                batch = map(lambda x: x.cuda(get_gpu_id()), batch) # EX9

            model.to(device)

            # Step 2 - forward pass: y' = model(x)
            outputs = model(inputs, lengths)  # EX9
            
            label = torch.FloatTensor([[1,0] if i == 0 else [0,1] for i in labels])
            # Step 3 - compute loss.
            # We compute the loss only for inspection (compare train/test loss)
            # because we do not actually backpropagate in test time
            loss = loss_function(outputs.to(device), label.to(device))  # EX9

            # Step 4 - make predictions (class = argmax of posteriors)
            _, predicted = torch.max(outputs,1)  # EX9

            # Step 5 - collect the predictions, gold labels and batch loss
            batch_size = list(inputs.size())[0] # EX9
            start = index * batch_size # EX9
            end = start + batch_size # EX9
            

            predicted = predicted.cpu() # EX9
            y_pred[start:end] = predicted.numpy() # EX9
            labels = labels.cpu() # EX9
            y[start:end] = labels.numpy()  # EX9

            running_loss += loss.data.item()

    return running_loss / index, (y_pred, y)
