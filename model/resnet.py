import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable 
from torchvision import datasets, transforms, models
from torch.utils.data import SubsetRandomSampler
import numpy as np
import helper



transform = transforms.Compose([
    #transforms.Scale((255, 255)),
    #transforms.Resize((100, 100)),
    transforms.Resize((225, 225)),
    #transforms.CenterCrop(225),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

dataset = datasets.ImageFolder("../data/images/", transform=transform)


# In[26]:


def get_subset(indices, start, end):
    return indices[start : start + end]


TRAIN_PCT, VALIDATION_PCT = 0.6, 0.2  # rest will go for test
train_count = int(len(dataset) * TRAIN_PCT)
validation_count = int(len(dataset) * TRAIN_PCT)

indices = torch.randperm(len(dataset))

train_indices = get_subset(indices, 0, train_count)
validation_indices = get_subset(indices, train_count, validation_count)
test_indices = get_subset(indices, train_count + validation_count, len(dataset))


# In[27]:


test_indices


# In[5]:


dataloaders = {
    "train": torch.utils.data.DataLoader(
        dataset, sampler=SubsetRandomSampler(train_indices),
        batch_size=32
    ),
    "validation": torch.utils.data.DataLoader(
        dataset, sampler=SubsetRandomSampler(validation_indices),
        batch_size=32
    ),
    "test": torch.utils.data.DataLoader(
        dataset, sampler=SubsetRandomSampler(test_indices),
        batch_size=32
    ),
}


# In[6]:


classes = ["abstract", "address", "affiliation", "author", "date", "email", "journal", "other", "title"]


# In[7]:


dataiter = iter(dataloaders['train'])
images, labels = next(dataiter)
images = images.numpy()
figure = plt.figure(figsize=(25, 10))

print(len(images))
for i in np.arange(5):
    ax = figure.add_subplot(2, 3, i + 1, xticks=[], yticks=[])
    image = images[i]/2+0.5 #unnormalize
    plt.imshow(image.reshape(225, -1))
    ax.set_title(classes[labels[i]])


# In[12]:


device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

model = models.resnet34(pretrained=True)

# Turn off gradients for our model
for param in model.parameters():
    param.requires_grad = False
    

# Define new classifier
classifier = nn.Sequential(nn.Linear(512, 128),
                      nn.ReLU(),
                      nn.Dropout(p=0.2),
                      nn.Linear(128, 10),
                      nn.LogSoftmax(dim = 1))

model.fc = classifier

criterion = nn.NLLLoss()

optimizer = optim.Adam(model.fc.parameters(), lr=0.003)

model.to(device)


# In[13]:


device 


# In[ ]:


epochs = 96
steps = 0
running_loss = 0
print_every = 10

for epoch in range(epochs):
    for images, labels in dataloaders['train']:
        steps +=1
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model(images)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            model.eval()
            test_loss = 0
            accuracy = 0
            
            for images, labels in dataloaders['validation']:
                
                images, labels = images.to(device), labels.to(device)
                
                logps = model(images)
                loss = criterion(logps, labels)
                test_loss += loss.item()
                
                ps = torch.exp(logps)
                top_ps, top_class = ps.topk(1, dim=1)
                
                equality= top_class ==labels.view(*top_class.shape)
                accuracy += torch.mean(equality.type(torch.FloatTensor))
            
            len_t = len(dataloaders["validation"])
            print(len_t)
            print(f"Epoch {epoch+9}/{epochs}.."
                  f"Train loss: {running_loss/print_every:.3f}.."
                  f"Test loss: {test_loss/len_t:.3f}.."
                  f"Test accuracy: {accuracy/len_t:.3f}"
                 )
            running_loss = 0
            model.train()
                


# In[ ]:





# In[31]:


nb_classes = 10

confusion_matrix = torch.zeros(nb_classes, nb_classes)
with torch.no_grad():
    for i, (inputs, classes) in enumerate(dataloaders['train']):
        inputs = inputs.to(device)
        classes = classes.to(device)
        logps = model(inputs)
        ps = torch.exp(logps)
        _, preds = torch.max(ps, 1)
        for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1


# In[ ]:


print(confusion_matrix.diag()/confusion_matrix.sum(1))


# In[ ]:




