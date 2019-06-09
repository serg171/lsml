import numpy as np
import os
import re
import torch
from tqdm import tqdm
import pandas as pd

def get_new_model_path(path, suffix=''):
    numered_runs = []
    for x in os.listdir(path):
        r = re.match('(\d+)', x)
        if r:
            numered_runs.append((os.path.join(path, x), int(r.group())))

    numered_runs.sort(key=lambda t: t[1])
    if len(numered_runs) == 0:
        new_number = 0
    else:
        _, nums = zip(*numered_runs)
        new_number = nums[-1] + 1
    if suffix != '':
        suffix = '_' + suffix
    t = os.path.join(path, '{}{}'.format(new_number, suffix))
    os.mkdir(t)
    os.mkdir(os.path.join(t, 'eval'))
    return t



def make_prediction(model, test_loader, IDs_to_sent, submit_name=''):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    model = model.to(device)
    model.eval()
    
    predictions = []
    for inputs in tqdm(test_loader):
        with torch.no_grad():
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, -1)
            preds = preds.to('cpu')
            predictions.extend(preds.data.numpy())
    predictions = np.array(predictions).reshape(-1, 1)
    IDs_to_sent = np.array(IDs_to_sent).reshape(-1, 1)
    result = np.hstack((IDs_to_sent, predictions))
    result = pd.DataFrame(result, columns=['id', 'predicted'])
    result.to_csv('/data/iNat/submit'+submit_name, index=False)
    
    