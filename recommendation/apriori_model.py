"""
Shopping Recommendation System Engine
=====================================
Implements Association Rule Mining using the Apriori Algorithm.
Supports both mlxtend library integration and a standalone educational
pure-Python Apriori engine for demonstration and resilience.

Key Metrics:
- Support(A -> B)    = count(A U B) / total_transactions
- Confidence(A -> B) = Support(A U B) / Support(A)
- Lift(A -> B)       = Confidence(A -> B) / Support(B)
"""

import os
import itertools
import pandas as pd
import numpy as np

# Try importing mlxtend; if available, use it; otherwise fallback gracefully
try:
    # pyrefly: ignore [missing-import]
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import apriori as mlx_apriori
    from mlxtend.frequent_patterns import association_rules as mlx_rules
    HAS_MLXTEND = True
except ImportError:
    HAS_MLXTEND = False


class AprioriRecommender:
    """
    Recommendation model based on Association Rule Mining.
    """

    def __init__(self, min_support=0.03, min_confidence=0.15, min_lift=1.0):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.rules_df = pd.DataFrame()
        self.frequent_itemsets = pd.DataFrame()
        self.transactions = []
        self.total_transactions = 0

    def load_transactions_from_csv(self, csv_path):
        """
        Loads transactions from CSV file (supports 'TransactionID,Items' where items are semicolon/comma separated).
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset not found at {csv_path}")

        df = pd.read_csv(csv_path)
        transactions = []
        if 'Items' in df.columns:
            for items_str in df['Items'].dropna():
                delimiter = ';' if ';' in str(items_str) else ','
                items = [item.strip() for item in str(items_str).split(delimiter) if item.strip()]
                if items:
                    transactions.append(items)
        elif 'Product' in df.columns and 'TransactionID' in df.columns:
            grouped = df.groupby('TransactionID')['Product'].apply(list)
            transactions = grouped.tolist()
        else:
            # Assume first column is items if separated
            for row in df.iloc[:, 0].dropna():
                delimiter = ';' if ';' in str(row) else ','
                items = [item.strip() for item in str(row).split(delimiter) if item.strip()]
                if items:
                    transactions.append(items)

        self.transactions = transactions
        self.total_transactions = len(transactions)
        return transactions

    def fit_from_transactions(self, transactions):
        """
        Preprocesses transactions and mines association rules.
        """
        self.transactions = [t for t in transactions if len(t) > 0]
        self.total_transactions = len(self.transactions)
        if self.total_transactions == 0:
            self.rules_df = pd.DataFrame()
            return self.rules_df

        if HAS_MLXTEND:
            try:
                return self._fit_mlxtend(self.transactions)
            except Exception as e:
                print(f"[Warning] mlxtend execution encountered issue: {e}. Falling back to internal Apriori algorithm.")
                return self._fit_standalone(self.transactions)
        else:
            return self._fit_standalone(self.transactions)

    def _fit_mlxtend(self, transactions):
        """Uses mlxtend library to compute Apriori and association rules."""
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

        # Mine frequent itemsets with Apriori
        frequent_itemsets = mlx_apriori(df_encoded, min_support=self.min_support, use_colnames=True)
        self.frequent_itemsets = frequent_itemsets

        if frequent_itemsets.empty:
            self.rules_df = pd.DataFrame(columns=['antecedents', 'consequents', 'support', 'confidence', 'lift'])
            return self.rules_df

        # Generate association rules
        rules = mlx_rules(frequent_itemsets, metric="confidence", min_threshold=self.min_confidence)
        if not rules.empty and self.min_lift > 0:
            rules = rules[rules['lift'] >= self.min_lift]

        # Convert frozensets to clean strings / lists for storage and display
        formatted_rules = []
        for _, row in rules.iterrows():
            ant_list = sorted(list(row['antecedents']))
            con_list = sorted(list(row['consequents']))
            formatted_rules.append({
                'antecedents': ", ".join(ant_list),
                'consequents': ", ".join(con_list),
                'antecedent_raw': ant_list,
                'consequent_raw': con_list,
                'support': round(float(row['support']), 4),
                'confidence': round(float(row['confidence']), 4),
                'lift': round(float(row['lift']), 4)
            })

        self.rules_df = pd.DataFrame(formatted_rules)
        if not self.rules_df.empty:
            self.rules_df.sort_values(by=['lift', 'confidence', 'support'], ascending=[False, False, False], inplace=True)
            self.rules_df.reset_index(drop=True, inplace=True)
        return self.rules_df

    def _fit_standalone(self, transactions):
        """
        Pure Python implementation of Apriori Algorithm.
        Ideal for explaining step-by-step to college viva/examiners!
        """
        n = len(transactions)
        item_counts = {}
        for t in transactions:
            for item in set(t):
                item_counts[frozenset([item])] = item_counts.get(frozenset([item]), 0) + 1

        # L1: Frequent 1-itemsets
        current_l = {itemset: count / n for itemset, count in item_counts.items() if (count / n) >= self.min_support}
        all_frequent = dict(current_l)
        k = 2

        while current_l:
            # Candidate generation C_k
            items_in_prev = list(current_l.keys())
            candidates = set()
            for i in range(len(items_in_prev)):
                for j in range(i + 1, len(items_in_prev)):
                    union_set = items_in_prev[i] | items_in_prev[j]
                    if len(union_set) == k:
                        # Apriori property check: all (k-1) subsets must be frequent
                        subsets = [union_set - frozenset([x]) for x in union_set]
                        if all(s in current_l for s in subsets):
                            candidates.add(union_set)

            # Count support for candidates
            candidate_counts = {c: 0 for c in candidates}
            for t in transactions:
                t_set = set(t)
                for c in candidates:
                    if c.issubset(t_set):
                        candidate_counts[c] += 1

            # Filter by min_support
            current_l = {c: count / n for c, count in candidate_counts.items() if (count / n) >= self.min_support}
            all_frequent.update(current_l)
            k += 1
            if k > 4:  # Keep max itemset size practical for mini project
                break

        # Generate Association Rules
        rules = []
        for itemset, sup_itemset in all_frequent.items():
            if len(itemset) >= 2:
                # Generate all non-empty proper subsets
                items = list(itemset)
                for r in range(1, len(items)):
                    for ant in itertools.combinations(items, r):
                        ant_set = frozenset(ant)
                        con_set = itemset - ant_set
                        sup_ant = all_frequent.get(ant_set, 0)
                        sup_con = all_frequent.get(con_set, 0)

                        if sup_ant > 0 and sup_con > 0:
                            confidence = sup_itemset / sup_ant
                            lift = confidence / sup_con
                            if confidence >= self.min_confidence and lift >= self.min_lift:
                                rules.append({
                                    'antecedents': ", ".join(sorted(list(ant_set))),
                                    'consequents': ", ".join(sorted(list(con_set))),
                                    'antecedent_raw': sorted(list(ant_set)),
                                    'consequent_raw': sorted(list(con_set)),
                                    'support': round(sup_itemset, 4),
                                    'confidence': round(confidence, 4),
                                    'lift': round(lift, 4)
                                })

        self.rules_df = pd.DataFrame(rules)
        if not self.rules_df.empty:
            # Deduplicate and sort
            self.rules_df.drop_duplicates(subset=['antecedents', 'consequents'], inplace=True)
            self.rules_df.sort_values(by=['lift', 'confidence', 'support'], ascending=[False, False, False], inplace=True)
            self.rules_df.reset_index(drop=True, inplace=True)
        return self.rules_df

    def get_recommendations_for_product(self, product_name, top_n=4):
        """
        Finds association rules where product_name is in the antecedent.
        Returns recommended items sorted by lift and confidence.
        """
        if self.rules_df.empty:
            return []

        matched_rules = []
        for _, row in self.rules_df.iterrows():
            ant_items = [i.strip() for i in row['antecedents'].split(',')]
            con_items = [i.strip() for i in row['consequents'].split(',')]

            # If user product is in antecedents, recommend the consequents
            if product_name.lower() in [i.lower() for i in ant_items]:
                for c in con_items:
                    if c.lower() != product_name.lower():
                        matched_rules.append({
                            'product_name': c,
                            'support': row['support'],
                            'confidence': row['confidence'],
                            'lift': row['lift'],
                            'reason': f"Rule: '{row['antecedents']}' → '{row['consequents']}' (Confidence: {row['confidence']*100:.1f}%, Lift: {row['lift']:.2f}x)"
                        })

        # Deduplicate recommendations taking the highest lift for each consequent
        seen = {}
        for r in matched_rules:
            p = r['product_name']
            if p not in seen or r['lift'] > seen[p]['lift']:
                seen[p] = r

        results = list(seen.values())
        results.sort(key=lambda x: (x['lift'], x['confidence']), reverse=True)
        return results[:top_n]

    def get_cart_recommendations(self, cart_product_names, top_n=4):
        """
        Given a list of product names currently in cart, finds multi-item or single-item
        association rules to recommend complementary products.
        """
        if self.rules_df.empty or not cart_product_names:
            return []

        cart_set = set(p.lower() for p in cart_product_names)
        matches = []

        for _, row in self.rules_df.iterrows():
            ant_items = [i.strip().lower() for i in row['antecedents'].split(',')]
            con_items = [i.strip() for i in row['consequents'].split(',')]

            # If cart contains all antecedent items (or any antecedent)
            ant_set = set(ant_items)
            is_subset = ant_set.issubset(cart_set)
            is_overlap = bool(ant_set.intersection(cart_set))

            if is_overlap:
                for c in con_items:
                    if c.lower() not in cart_set:
                        priority_bonus = 2.0 if is_subset else 1.0
                        matches.append({
                            'product_name': c,
                            'support': row['support'],
                            'confidence': row['confidence'],
                            'lift': row['lift'],
                            'score': row['lift'] * priority_bonus,
                            'reason': f"Frequently bought with items in your cart (Lift: {row['lift']:.2f}x)"
                        })

        seen = {}
        for m in matches:
            p = m['product_name']
            if p not in seen or m['score'] > seen[p]['score']:
                seen[p] = m

        results = list(seen.values())
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_n]

    def get_summary_statistics(self):
        """
        Returns statistical summary of data mining metrics for dashboard.
        """
        total_rules = len(self.rules_df)
        avg_support = float(self.rules_df['support'].mean()) if total_rules > 0 else 0.0
        avg_confidence = float(self.rules_df['confidence'].mean()) if total_rules > 0 else 0.0
        avg_lift = float(self.rules_df['lift'].mean()) if total_rules > 0 else 0.0
        max_lift = float(self.rules_df['lift'].max()) if total_rules > 0 else 0.0

        top_rules = []
        if total_rules > 0:
            top_rules = self.rules_df.head(10).to_dict(orient='records')

        return {
            'total_rules': total_rules,
            'total_transactions': self.total_transactions,
            'avg_support': round(avg_support, 4),
            'avg_confidence': round(avg_confidence, 4),
            'avg_lift': round(avg_lift, 4),
            'max_lift': round(max_lift, 4),
            'top_rules': top_rules
        }
