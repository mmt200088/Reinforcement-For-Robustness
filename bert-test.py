
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import torch
from sklearn.metrics import accuracy_score

# 加载模型和分词器
model_name = "textattack/bert-base-uncased-MRPC"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 加载MRPC数据集
dataset = load_dataset("glue", "mrpc")
val_data = dataset["validation"]

# 数据预处理函数
def preprocess(examples):
    return tokenizer(
        examples["sentence1"],
        examples["sentence2"],
        padding="max_length",
        max_length=128
    )

# 处理验证集
val_data = val_data.map(preprocess, batched=True)
val_data.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "token_type_ids", "label"]
)
# 评估函数
def evaluate():
    model.eval()
    predictions = []
    true_labels = []
    
    for batch in val_data:
        with torch.no_grad():
            outputs = model(
                input_ids=batch["input_ids"].unsqueeze(0),
                attention_mask=batch["attention_mask"].unsqueeze(0),
                 token_type_ids=batch["token_type_ids"].unsqueeze(0)
            )
            logits = outputs.logits
            pred = torch.argmax(logits, dim=-1).item()
            predictions.append(pred)
            true_labels.append(batch["label"].item())
    
    accuracy = accuracy_score(true_labels, predictions)
    print(f"Validation Accuracy: {accuracy:.4f}")

# 执行评估
evaluate()
