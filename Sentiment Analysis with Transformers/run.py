import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from matplotlib import pyplot as plt

MODEL = "microsoft/xtremedistil-l6-h256-uncased"
EPOCHS = 4
BATCH_SIZE = 8
LEARNING_RATE = 3e-5
MAX_SIZE = 256
dataset = load_dataset("rotten_tomatoes")
torch.manual_seed(123)

tokenizer = AutoTokenizer.from_pretrained("microsoft/xtremedistil-l6-h256-uncased")
model = AutoModelForSequenceClassification.from_pretrained("microsoft/xtremedistil-l6-h256-uncased", num_labels=2)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# 1.1 Preprocessing
preprocessed_data = dataset.map(lambda examples: tokenizer(examples['text'],
                                                           padding='max_length', truncation=True, max_length=256),
                                batched=True)

train = preprocessed_data['train'].shuffle().select(range(2000))
train.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])

val = preprocessed_data['validation'].shuffle().select(range(200))
val.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])

test = preprocessed_data['test'].shuffle().select(range(200))
test.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])

train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val, batch_size=BATCH_SIZE)
test_loader = DataLoader(test, batch_size=BATCH_SIZE)

# Initialize track lists and "best" score 
loss_hist = []
acc_hist = []
f1_hist = []
best = 0

# 1.2 Training and Validation

# (tqdm isn't necessary, but provides a nice progress bar)
for i in range(EPOCHS):

    # Training
    model.train()
    t_train = tqdm(train_loader, position=0, leave=True, desc="Epoch: {}".format(i))
    for batch in t_train:
        optimizer.zero_grad()
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                        token_type_ids=batch['token_type_ids'], labels=batch['label'])
        loss = outputs.loss.item()
        outputs.loss.backward()
        optimizer.step()
        loss_hist.append(loss)
        t_train.set_postfix(loss=loss, refresh=False)
        t_train.refresh()

    # Validation
    model.eval()
    predictions = []
    labels = []
    t_val = tqdm(val_loader, position=0, leave=True, desc="Validating ...")
    for batch in t_val:
        with torch.no_grad():
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                            token_type_ids=batch['token_type_ids'], labels=batch['label'])
            # calculate model predictions
            predictions.extend(torch.argmax(outputs[1], axis=1).tolist())
            labels.extend(batch['label'].tolist())
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    acc_hist.append(acc)
    f1_hist.append(f1)
    print("Epoch: {}, Accuracy: {:.4f}, F1-Score: {:.4f}".format(i, acc, f1))
    if best < f1:
        best = f1
        model.save_pretrained("best_model")

# Plots
fig, ax = plt.subplots(2, 1, tight_layout=True, figsize=(9,16))
ax[0].plot(loss_hist)
ax[0].set_title("Loss History")
ax[0].set_ylabel("Loss")
ax[0].set_xlabel("Training Step")
ax[1].plot(acc_hist, label='Accuracy')
ax[1].plot(f1_hist, label='F1 Score')
ax[1].set_title("Accuracy / F1 Score on Dev Set")
ax[1].set_ylabel("Accuracy / F1")
ax[1].set_xlabel("# of Epoch")
ax[1].legend(loc='lower right')
plt.savefig("plots_1_4_cpu.png", dpi=300)
plt.show()

# Testing
model.from_pretrained("best_model")
model.eval()
predictions = []
labels = []
with torch.no_grad():
    for batch in tqdm(test_loader, position=0, leave=True, desc="Testing ..."):
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                            token_type_ids=batch['token_type_ids'], labels=batch['label'])
        # calculate model predictions
        predictions.extend(torch.argmax(outputs[1], axis=1).tolist())
        labels.extend(batch['label'].tolist())
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    print("Test set: Accuracy: {:.4f}, F1-Score: {:.4f}".format(acc, f1))