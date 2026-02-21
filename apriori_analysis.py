import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


# Load dataset
df = pd.read_csv('Online_Courses.csv')


# Prepare skill transactions (remove empty strings)
transactions = df['Skills'].dropna().apply(
    lambda x: [i.strip() for i in x.split(',') if i.strip() != '']
).tolist()


# Print sample transactions
print("Sample transactions:")
for t in transactions[:5]:
    print(t)


# One-hot encode transactions
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
onehot_df = pd.DataFrame(te_ary, columns=te.columns_)


# Print skill counts
print("\nSkill counts:")
print(onehot_df.sum().sort_values(ascending=False))


# Use lower support threshold to find more frequent itemsets
frequent_itemsets = apriori(onehot_df, min_support=0.01, use_colnames=True)


# No filtering on length here
freq_itemsets_filtered = frequent_itemsets


# Generate association rules with confidence threshold
rules = association_rules(freq_itemsets_filtered, metric="confidence", min_threshold=0.2)


# Safeguard: check if rules DataFrame is empty before filtering
if rules.empty:
    print("No association rules found with current support/confidence settings.")
else:
    # Remove rules with empty antecedents or consequents
    rules = rules[rules['antecedents'].apply(lambda x: len(x) > 0)]
    rules = rules[rules['consequents'].apply(lambda x: len(x) > 0)]


    print("\nAssociation Rules:")
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
