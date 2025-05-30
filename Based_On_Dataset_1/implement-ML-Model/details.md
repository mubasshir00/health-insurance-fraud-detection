##

## Data Set - https://www.kaggle.com/datasets/nyashachizampeni/medical-insurance-claim-fraud/data


## 🌲 How Random Forest Detects Fraud – With Example

Random Forest is a machine learning algorithm that builds **many decision trees**, each trained on different parts of the dataset. The final prediction is made by **majority vote** of these trees.

---

### 📋 Sample Data

| Fee Charged | Membership Period | Cause                | Label        |
|-------------|-------------------|----------------------|--------------|
| 10,000      | 3,000 days        | Road Traffic Accident| 0 (Clean)    |
| 50,000      | 200 days          | Other                | 1 (Fraud)    |
| 12,000      | 4,000 days        | Work Accident        | 0 (Clean)    |
| 70,000      | 150 days          | Other                | 1 (Fraud)    |

---

### 🌳 What a Single Decision Tree Might Learn

```plaintext
IF Fee Charged > 40,000
    AND Membership Period < 1000
    AND Cause == "Other"
→ THEN label = 1 (Fraud)
ELSE → label = 0 (Clean)
```

✅ In the table above:
- Rows 2 and 4 match → Fraud
- Rows 1 and 3 don’t → Clean

---

### 🌲 What Random Forest Does

It builds **many different trees**, each making independent decisions. Then:

- **Tree 1**: votes Fraud
- **Tree 2**: votes Clean
- **Tree 3**: votes Fraud

✅ **Majority Vote** → Final Prediction = **Fraud**

---

### 🧠 Real-Life Example from Your Data

| gender | location | cause | Fee Charged | membership_period |
|--------|----------|-------|-------------|-------------------|
| female | Harare   | Other | 85,000      | 250               |

- Tree A: suspicious cause + high fee + short membership → Fraud
- Tree B: high fee alone → Fraud
- Tree C: not sure → Clean

✅ 2 out of 3 trees vote **Fraud**

```python
model.predict([this_row]) → [1]  # 1 means Fraudulent
```

---

### 🔍 Does Random Forest "Know" Fraud?

No, it doesn’t know what fraud *is*.  
It simply **learns patterns** from labeled data:

> “Claims with high amounts + short membership + vague causes often = fraud”
