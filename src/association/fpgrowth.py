from collections import defaultdict
import pandas as pd

class FPNode:
    """Single node in an FP-Tree."""
    __slots__ = ("item", "count", "parent", "children", "node_link")

    def __init__(self, item, count: int = 0, parent=None):
        self.item = item
        self.count = count
        self.parent = parent
        self.children: dict = {}
        self.node_link = None

    def increment(self, count: int = 1):
        self.count += count

class FPTree:
    """
    FP-Tree built from a list of transactions.
    """

    def __init__(self):
        self.root = FPNode(item=None, count=0)
        self.header_table: dict[str, list] = {}   # item -> [count, first_node]

    def insert_transaction(self, transaction: list, count: int = 1):
        """Insert a single (pre-sorted) transaction into the tree."""
        node = self.root
        for item in transaction:
            if item not in self.header_table:
                self.header_table[item] = [0, None]
            self.header_table[item][0] += count
 
            if item in node.children:
                node.children[item].increment(count)
            else:
                new_node = FPNode(item=item, count=count, parent=node)
                node.children[item] = new_node
                # Append to node-link chain
                if self.header_table[item][1] is None:
                    self.header_table[item][1] = new_node
                else:
                    cur = self.header_table[item][1]
                    while cur.node_link is not None:
                        cur = cur.node_link
                    cur.node_link = new_node
            node = node.children[item]

    def is_single_path(self) -> bool:
        """Return True if the tree consists of a single path (no branching)."""
        node = self.root
        while node.children:
            if len(node.children) > 1:
                return False
            node = next(iter(node.children.values()))
        return True

    def get_path_nodes(self) -> list:
        """Return ordered list of nodes along the single path (root excluded)."""
        path = []
        node = self.root
        while node.children:
            node = next(iter(node.children.values()))
            path.append(node)
        return path

    def prefix_paths(self, item) -> list:
        """
        Return all prefix paths ending just above each node with `item`.
        Each path is returned as list of (item, count) tuples.
        """
        paths = []
        node = self.header_table[item][1]
        while node is not None:
            prefix = []
            ancestor = node.parent
            while ancestor.item is not None:
                prefix.append(ancestor.item)
                ancestor = ancestor.parent
            prefix.reverse()
            if prefix:
                paths.append((prefix, node.count))
            node = node.node_link
        return paths

#  Core FP-Growth recursive miner
def _fp_growth_recursive(
    tree: FPTree,
    min_support_count: int,
    prefix: frozenset,
    frequent_itemsets: dict,
):
    """
    Recursive worker.  Mines `tree` and writes results into `frequent_itemsets`.
    """
    # Single-path optimisation
    if tree.is_single_path():
        path_nodes = tree.get_path_nodes()
        # Enumerate all non-empty subsets of path nodes
        for r in range(1, len(path_nodes) + 1):
            for combo in _combinations_iter(path_nodes, r):
                itemset = prefix | frozenset(n.item for n in combo)
                # Support = minimum count along the chosen nodes
                support_count = min(n.count for n in combo)
                if support_count >= min_support_count:
                    frequent_itemsets[itemset] = support_count
        return

    # General case: iterate header table in ascending support order
    items_by_support = sorted(
        tree.header_table.keys(),
        key=lambda x: tree.header_table[x][0],
    )

    for item in items_by_support:
        item_support = tree.header_table[item][0]
        if item_support < min_support_count:
            continue

        new_prefix = prefix | frozenset([item])
        frequent_itemsets[new_prefix] = item_support

        # Build conditional pattern base
        cond_transactions = []
        for path, count in tree.prefix_paths(item):
            cond_transactions.append((path, count))

        # Build conditional FP-Tree
        cond_tree = _build_conditional_tree(
            cond_transactions, min_support_count
        )
        if cond_tree.header_table:
            _fp_growth_recursive(
                cond_tree, min_support_count, new_prefix, frequent_itemsets
            )

def _build_conditional_tree(
    cond_transactions: list,   # list of (path: list[str], count: int)
    min_support_count: int,
) -> FPTree:
    """Build a conditional FP-Tree from conditional pattern base."""
    # Count item frequencies in the conditional base
    item_counts: dict[str, int] = defaultdict(int)
    for path, count in cond_transactions:
        for item in path:
            item_counts[item] += count

    # Keep only frequent items
    frequent_items = {i for i, c in item_counts.items() if c >= min_support_count}

    cond_tree = FPTree()
    for path, count in cond_transactions:
        filtered = [i for i in path if i in frequent_items]
        # Sort descending by frequency (same ordering strategy as root tree)
        filtered.sort(key=lambda x: item_counts[x], reverse=True)
        if filtered:
            cond_tree.insert_transaction(filtered, count)

    return cond_tree

def _combinations_iter(items, r):
    """Yield all r-combinations of items (list)."""
    from itertools import combinations as _comb
    return _comb(items, r)


def custom_fpgrowth(transactions: list, min_support: float) -> dict:
    """
    Custom FP-Growth implementation.
    Returns: dict {tuple_of_sorted_items: support_ratio}
    """
    num_transactions = len(transactions)
    min_support_count = min_support * num_transactions

    # Count 1-item frequencies
    item_counts: dict[str, int] = defaultdict(int)
    for t in transactions:
        for item in t:
            item_counts[item] += 1

    frequent_items = {
        item for item, cnt in item_counts.items()
        if cnt >= min_support_count
    }

    # Build root FP-Tree
    tree = FPTree()
    for t in transactions:
        # Filter and sort items: descending frequency, then lexicographic
        filtered = sorted(
            (item for item in t if item in frequent_items),
            key=lambda x: (-item_counts[x], x),
        )
        if filtered:
            tree.insert_transaction(filtered)

    # Mine frequent itemsets
    frequent_itemsets: dict[frozenset, int] = {}

    # Add frequent 1-itemsets explicitly
    for item, cnt in item_counts.items():
        if cnt >= min_support_count:
            frequent_itemsets[frozenset([item])] = cnt

    _fp_growth_recursive(tree, min_support_count, frozenset(), frequent_itemsets)

    # Convert to support ratios with sorted tuple keys
    return {
        tuple(sorted(k)): v / num_transactions
        for k, v in frequent_itemsets.items()
    }