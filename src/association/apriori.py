from itertools import chain, combinations
import pandas as pd

def get_subsets(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)))

def generate_custom_rules(frequent_itemsets, min_confidence=0.5, min_lift=1.0):
    """
    Frequent_itemsets: dict {tuple_of_items: support_value}
    """
    rules = []
    
    for itemset, support in frequent_itemsets.items():
        if len(itemset) > 1:
            for antecedent in get_subsets(itemset):
                antecedent = tuple(sorted(antecedent))
                consequent = tuple(sorted(set(itemset) - set(antecedent)))
                
                support_A = frequent_itemsets.get(antecedent, 0)
                support_B = frequent_itemsets.get(consequent, 0)
                
                if support_A == 0 or support_B == 0:
                    continue
                    
                confidence = support / support_A
                lift = confidence / support_B
                
                if confidence >= min_confidence and lift >= min_lift:
                    rules.append({
                        'Antecedents': ", ".join(antecedent),
                        'Consequents': ", ".join(consequent),
                        'Support': support,
                        'Confidence': confidence,
                        'Lift': lift
                    })
    
    if not rules:
        return pd.DataFrame(columns=['Antecedents', 'Consequents', 'Support', 'Confidence', 'Lift'])

    return pd.DataFrame(rules).sort_values(by='Lift', ascending=False).reset_index(drop=True)

def custom_apriori(transactions, min_support):
    """
    Custom implementation of Apriori algorithm.
    Returns a dictionary of frequent itemsets and their support.
    """
    num_transactions = len(transactions)
    min_support_count = min_support * num_transactions
    
    # Convert transactions to list of sets for fast subset operations
    transaction_sets = [set(t) for t in transactions]
    
    # Count 1-itemsets
    item_counts = {}
    for t in transaction_sets:
        for item in t:
            item_counts[frozenset([item])] = item_counts.get(frozenset([item]), 0) + 1
            
    # Filter L1 (Frequent 1-itemsets)
    current_l = {itemset: count for itemset, count in item_counts.items() if count >= min_support_count}
    all_frequent_itemsets = current_l.copy()
    
    k = 2
    while current_l:
        # Generate candidates C_k from L_{k-1}
        l_keys = list(current_l.keys())
        candidates = set()
        for i in range(len(l_keys)):
            for j in range(i + 1, len(l_keys)):
                # Union of two sets
                candidate = l_keys[i] | l_keys[j]
                if len(candidate) == k:
                    candidates.add(candidate)
                    
        # Count support for candidates
        candidate_counts = {c: 0 for c in candidates}
        for t in transaction_sets:
            for c in candidates:
                if c.issubset(t):
                    candidate_counts[c] += 1
                    
        # Filter L_k
        current_l = {c: count for c, count in candidate_counts.items() if count >= min_support_count}
        all_frequent_itemsets.update(current_l)
        k += 1
        
    # Convert counts to support ratio
    result = {tuple(k): v / num_transactions for k, v in all_frequent_itemsets.items()}
    return result