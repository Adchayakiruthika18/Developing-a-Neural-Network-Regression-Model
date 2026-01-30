# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model
<img width="1095" height="745" alt="Screenshot 2026-01-30 141322" src="https://github.com/user-attachments/assets/b2141eb2-ad6c-4f3d-8419-c8126c9732b7" />

## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name:Adchayakiruthika M S

### Register Number:212223230005

```python
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(1,8)
        self.fc2=nn.Linear(8,10)
        self.fc3=nn.Linear(10,1)
        self.relu=nn.ReLU()
        self.history={'loss':[]}
    def forward(self,x):
        x=self.relu(self.fc1(x))
        x=self.relu(self.fc2(x))
        x=self.fc3(x)
        return x



# Initialize the Model, Loss Function, and Optimizer
ai_brain=NeuralNet()
criterion=nn.MSELoss()
optimizer=optim.RMSprop(ai_brain.parameters(),lr=0.001)




def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
      optimizer.zero_grad()
      Loss=criterion(ai_brain(X_train),y_train)
      Loss.backward()
      optimizer.step()
      ai_brain.history['loss'].append(Loss.item())
      if epoch % 200 == 0:
        print(f'Epoch [{epoch}/{epochs}], Loss: {Loss.item():.6f}')


```

### Dataset Information
<img width="213" height="287" alt="Screenshot 2026-01-30 141902" src="https://github.com/user-attachments/assets/e71a5455-26d7-4039-98b7-249c64f7ab73" />

### OUTPUT

### Loss

<img width="494" height="238" alt="Screenshot 2026-01-30 142057" src="https://github.com/user-attachments/assets/1d17e49a-5204-46ab-bd96-a268d7f93a14" />

<img width="422" height="39" alt="Screenshot 2026-01-30 142129" src="https://github.com/user-attachments/assets/50270468-31a9-4b31-8b6b-0dc970745259" />

### Training Loss Vs Iteration Plot
<img width="728" height="577" alt="Screenshot 2026-01-30 142207" src="https://github.com/user-attachments/assets/83a7d8eb-98d1-4b97-8af1-51dfc009f119" />

### New Sample Data Prediction
<img width="468" height="40" alt="Screenshot 2026-01-30 142245" src="https://github.com/user-attachments/assets/392894c0-79b7-4704-bd1f-bba6a66e595d" />

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
