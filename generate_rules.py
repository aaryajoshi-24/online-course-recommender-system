import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Load your courses CSV
df = pd.read_csv('Online_Courses.csv')

# Process course skills into transactions list
transactions = df['Skills'].dropna().apply(lambda x: [i.strip() for i in x.split(',') if i.strip() != '']).tolist()

# One-hot encode the transactions
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
onehot_df = pd.DataFrame(te_ary, columns=te.columns_)

# Find frequent itemsets with minimum support 1%
frequent_itemsets = apriori(onehot_df, min_support=0.01, use_colnames=True)

# Generate association rules with minimum confidence 20%
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2)

# Save rules to CSV
rules.to_csv('association_rules.csv', index=False)

print("association_rules.csv file has been created!")
