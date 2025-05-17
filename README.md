# Multi-Label Emotion Recognition from Text

Hi there!  
This is my end-to-end Machine Learning project for detecting **multiple emotions** in a single text input using **Google's GoEmotions dataset** and **BERT (a transformer-based model)**.

## Objective

To build a system that can classify one or more emotions (like joy, anger, sadness, etc.) from a sentence or short paragraph. This is useful in areas like:
- Analyzing social media sentiment
- Understanding customer feedback
- Chatbot emotional intelligence

## Dataset

I used the **GoEmotions Dataset** released by Google AI:
- Contains over 58,000 carefully labeled Reddit comments
- Supports **multi-label emotion classification**
- 28 emotion classes + a neutral class

## Project Pipeline

Here’s how I built this project:

### 1. Data Preprocessing
- Loaded the GoEmotions dataset using the HuggingFace `datasets` library
- Removed unnecessary columns
- Converted label indices to multi-hot vectors
- (Class imbalance handling planned for future improvements!)

### 2. Tokenization
- Used a pre-trained **BERT tokenizer** to tokenize input text
- Made sure labels were formatted as float32 for loss function compatibility

### 3. Model Training
- Fine-tuned a **`bert-base-uncased`** model using `Trainer` from HuggingFace
- Trained using binary cross-entropy loss
- Used evaluation metrics: **Hamming Loss** and **Micro F1 Score**

### 4. Model Evaluation
Here are the final results from 5 epochs:

| Epoch | F1 Score | Hamming Loss |
|-------|----------|---------------|
| 1     | 0.405    | 0.0986        |
| 2     | 0.412    | 0.0977        |
| 3     | 0.440    | 0.0854        |
| 4     | 0.472    | 0.0739        |
| 5     | 0.472    | 0.0728        |

*You can see the model improved steadily and learned to handle multiple emotions per text sample.*

### 5. Model Saving
- The trained model is saved as `goemotions_model/` (includes tokenizer and config)
- You can download and reload it for future predictions

### 6. Testing on Real Text
Try things like:

```python
text = "I’m so happy and proud of myself!"
model.predict([text])
# Output: ['joy', 'pride']
```

---

## How to Run

### Install Requirements
```bash
pip install -r requirements.txt
```

### Run Notebook
- Open the notebook `GoEmotions_Multi_Label_Classification.ipynb`
- Follow the steps: preprocessing → training → saving → testing

---

## Files Included

- `GoEmotions_Multi_Label_Classification.ipynb` – Full training code
- `requirements.txt` – Python dependencies

---

## About Me

I’m currently learning machine learning and this is one of my end-to-end projects involving **transformers**, **multi-label classification**, and **real-world deployment** workflows.

If you found this useful or have feedback, feel free to open an issue or reach out!

---

### To-Do / Future Work

- Handle class imbalance using techniques like oversampling or class weights
- Deploy as a REST API for real-time emotion detection
- Create a simple front-end UI

---

Thanks for checking it out! 
*— Tahir Ali*
