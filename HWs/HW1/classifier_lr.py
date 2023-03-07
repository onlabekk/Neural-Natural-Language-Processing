class UnsafeDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.LongTensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

class LogRegModel():
    def __init__(self, vocab_size, num_classes, embed_dim):
        super(LogRegModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, data):
        embedded = self.embedding(data)
        return self.fc(embedded)
    
def train_model_epoch(model, opt, loss_fn, dataloader, writer, begin):
    model.train()
    train_loss = 0
    pred = []
    act = []
    step = begin

    for sample in tqdm_notebook(dataloader):
        opt.zero_grad()

        ids = sample["input_ids"].to(device)
        target = sample["labels"].to(device)
        out = model(ids)
        loss = loss_fn(out, target)
        train_loss += loss.item()
        loss.backward()
        opt.step()

        pred_cls = F.softmax(out, dim=1).argmax(1)
        pred_cls = pred_cls.squeeze().detach().cpu().numpy().tolist()
        pred.extend(pred_cls)
        act.extend(target.squeeze().cpu().numpy().tolist())
        writer.add_scalar('Loss/total', loss.item(), step)
        step += 1

    acc = accuracy_score(pred, act)
    f1 = f1_score(pred, act)
    precision = precision_score(pred, act)
    recall = recall_score(pred, act)
    writer.add_scalar('Loss/train', train_loss / len(dataloader.dataset), epoch)
    writer.add_scalar('Accuracy/train', acc, epoch)
    writer.add_scalar('F1_score/train', f1, epoch)
    writer.add_scalar('Precision/train', precision, epoch)

    return train_loss / len(dataloader.dataset), {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }, step

def val_model_epoch(model, dataloader, writer):
    model.eval()
    pred = []
    act = []

    for sample in tqdm_notebook(dataloader):
        ids = sample["input_ids"].to(device)
        target = sample["labels"].to(device)
        out = model(ids)
        pred_cls = F.softmax(out, dim=1).argmax(1)
        pred_cls = pred_cls.squeeze().detach().cpu().numpy().tolist()
        pred.extend(pred_cls)
        act.extend(target.squeeze().cpu().numpy().tolist())

    acc = accuracy_score(pred, act)
    f1 = f1_score(pred, act)
    precision = precision_score(pred, act)
    recall = recall_score(pred, act)
    writer.add_scalar('Accuracy/val', acc, epoch)
    writer.add_scalar('F1_score/val', f1, epoch)
    writer.add_scalar('Precision/val', precision, epoch)

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
 