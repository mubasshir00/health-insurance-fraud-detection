# Rule-based Fraud Detection

The dataset contains :

- `Fee Charged` : amount billed
- `membership_period` : total prior claims
- `number_of_claims` : total prior claims
- `number_of_dependants` : count of covered dependents
- `label` : 1 for fraud , 0 for legitimate

## Key Steps

### 1. Load and Prepare Data
- Read in the dataset
- Separate fraud (`label == 1`) and legitimate claims (`label == 0`)

### 2. Generate simple one-feature rules
- For each numeric field:
    - If fraud values are higher -> flag top 10% of legit values
    - If fraud values are lower -> flag bottom 10% of legit values
- Generate candidate rules like:
    - `Fee Charged > 45109`
    - `membership_period < 999`

### 3. Evaluate Rules
Each rule is evaluated using:
- **Precision:** Proportion of detected frauds that are actually fraud  
- **Recall:** Proportion of total frauds that were caught  
- **Count:** Total number of claims matched by the rule


## 📊 Metric Summary

| Metric     | Meaning |
|------------|---------|
| **Precision** | Among the claims predicted as fraud, how many are truly fraud? (TP / (TP + FP)) |
| **Recall**    | Among all actual frauds, how many did the rule catch? (TP / (TP + FN)) |
| **Count**     | Total number of claims flagged by the rule (TP + FP) |

TP = true positive
FP = false positive
FN = false negative