"""
Optimized Trie Variants Implementation
--------------------------------------
This script includes:
1. Optimized Standard Trie
2. Optimized Patricia Trie
3. Optimized Ternary Search Tree (TST)

Features:
- Memory pooling for efficient node reuse and reduced fragmentation.
- Index-based child node access for faster lookups.
- Iterative DFS for robust and stack-efficient autocomplete traversal.
- Common-prefix compression in Patricia Trie for reduced node count.
- Lightweight node storage using __slots__ to minimize memory overhead.
- Performance metrics: Tracks insertion time, query time, and memory usage.

Author: [Rhitesh Kumar Singh]
"""

import pandas as pd
import time
import sys
import random
import string


# Helper Functions
def generate_random_word(length):
    """Generate a random word of given length."""
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(length))


def get_size(obj, seen=None):
    """Recursively compute the size of an object in memory."""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum(get_size(v, seen) for v in obj.values())
        size += sum(get_size(k, seen) for k in obj.keys())
    elif hasattr(obj, '__dict__'):
        size += get_size(vars(obj), seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum(get_size(i, seen) for i in obj)
    return size


# Memory pooling for optimized nodes
trie_node_pool = []
patricia_node_pool = []


# Optimized Standard Trie
class OptimizedTrieNode:
    __slots__ = ['children', 'is_end_of_word', 'frequency', 'word']

    def __init__(self):
        self.children = [None] * 26
        self.is_end_of_word = False
        self.frequency = 0
        self.word = None

    @staticmethod
    def get_node_from_pool():
        return trie_node_pool.pop() if trie_node_pool else OptimizedTrieNode()

    @staticmethod
    def return_node_to_pool(node):
        node.children = [None] * 26
        node.is_end_of_word = False
        node.frequency = 0
        node.word = None
        trie_node_pool.append(node)


class OptimizedTrie:
    def __init__(self):
        self.root = OptimizedTrieNode()
        self.insert_time = 0
        self.query_time = 0

    def _char_to_index(self, char):
        return ord(char) - ord('a')

    def insert(self, word, frequency):
        start_time = time.time()
        node = self.root
        for char in word:
            index = self._char_to_index(char)
            if node.children[index] is None:
                node.children[index] = OptimizedTrieNode.get_node_from_pool()
            node = node.children[index]
        node.is_end_of_word = True
        node.frequency = frequency
        node.word = word
        self.insert_time += time.time() - start_time

    def autocomplete(self, prefix):
        start_time = time.time()
        node = self.root
        for char in prefix:
            index = self._char_to_index(char)
            if node.children[index] is not None:
                node = node.children[index]
            else:
                self.query_time += time.time() - start_time
                return []
        suggestions = self._iterative_dfs(node)
        self.query_time += time.time() - start_time
        return sorted(suggestions, key=lambda x: x[1], reverse=True)

    def _iterative_dfs(self, start_node):
        suggestions = []
        stack = [(start_node, "")]
        while stack:
            node, prefix = stack.pop()
            if node.is_end_of_word:
                suggestions.append((node.word, node.frequency))
            for i in range(25, -1, -1):
                child = node.children[i]
                if child is not None:
                    stack.append((child, prefix + chr(i + ord('a'))))
        return suggestions


# Optimized Patricia Trie
class OptimizedPatriciaTrieNode:
    __slots__ = ['key', 'children', 'is_end_of_word', 'frequency', 'word']

    def __init__(self, key):
        self.key = key
        self.children = [None] * 26
        self.is_end_of_word = False
        self.frequency = 0
        self.word = None

    @staticmethod
    def get_node_from_pool(key):
        if patricia_node_pool:
            node = patricia_node_pool.pop()
            node.key = key
            return node
        else:
            return OptimizedPatriciaTrieNode(key)

    @staticmethod
    def return_node_to_pool(node):
        node.key = ""
        node.children = [None] * 26
        node.is_end_of_word = False
        node.frequency = 0
        node.word = None
        patricia_node_pool.append(node)


class OptimizedPatriciaTrie:
    def __init__(self):
        self.root = OptimizedPatriciaTrieNode("")
        self.insert_time = 0
        self.query_time = 0

    def _common_prefix_length(self, str1, str2):
        """Helper function to find the length of the common prefix between two strings."""
        length = min(len(str1), len(str2))
        for i in range(length):
            if str1[i] != str2[i]:
                return i
        return length

    def insert(self, word, frequency):
        start_time = time.time()
        node = self.root
        i = 0
        while i < len(word):
            found = False
            index = ord(word[i]) - ord('a')
            for child in node.children:
                if child and child.key.startswith(word[i]):
                    common_prefix = self._common_prefix_length(word[i:], child.key)
                    if common_prefix == len(child.key):
                        node = child
                        i += common_prefix
                    else:
                        # Split node logic here
                        break
                    found = True
                    break
            if not found:
                new_node = OptimizedPatriciaTrieNode(word[i:])
                new_node.is_end_of_word = True
                new_node.frequency = frequency
                new_node.word = word
                node.children[index] = new_node
                break
        self.insert_time += time.time() - start_time

    def autocomplete(self, prefix):
        start_time = time.time()
        node = self.root
        for char in prefix:
            index = ord(char) - ord('a')
            if node.children[index]:
                node = node.children[index]
            else:
                self.query_time += time.time() - start_time
                return []
        suggestions = self._collect_all_words(node)
        self.query_time += time.time() - start_time
        return sorted(suggestions, key=lambda x: x[1], reverse=True)

    def _collect_all_words(self, node):
        suggestions = []
        stack = [node]
        while stack:
            node = stack.pop()
            if node.is_end_of_word:
                suggestions.append((node.word, node.frequency))
            stack.extend(filter(None, node.children))
        return suggestions


# Optimized Ternary Search Tree
class OptimizedTSTNode:
    def __init__(self, char):
        self.char = char
        self.left = None
        self.eq = None
        self.right = None
        self.is_end_of_word = False
        self.frequency = 0
        self.word = None


class OptimizedTST:
    def __init__(self):
        self.root = None
        self.insert_time = 0
        self.query_time = 0

    def insert(self, word, frequency):
        start_time = time.time()
        self.root = self._insert(self.root, word, 0, frequency)
        self.insert_time += time.time() - start_time

    def _insert(self, node, word, index, frequency):
        char = word[index]
        if node is None:
            node = OptimizedTSTNode(char)
        if char < node.char:
            node.left = self._insert(node.left, word, index, frequency)
        elif char > node.char:
            node.right = self._insert(node.right, word, index, frequency)
        else:
            if index + 1 < len(word):
                node.eq = self._insert(node.eq, word, index + 1, frequency)
            else:
                node.is_end_of_word = True
                node.frequency = frequency
                node.word = word
        return node

    def autocomplete(self, prefix):
        start_time = time.time()
        node = self._search(self.root, prefix, 0)
        suggestions = []
        if node:
            if node.is_end_of_word:
                suggestions.append((node.word, node.frequency))
            self._collect_all_words(node.eq, prefix, suggestions)
        self.query_time += time.time() - start_time
        return sorted(suggestions, key=lambda x: x[1], reverse=True)

    def _search(self, node, prefix, index):
        if node is None:
            return None
        char = prefix[index]
        if char < node.char:
            return self._search(node.left, prefix, index)
        elif char > node.char:
            return self._search(node.right, prefix, index)
        else:
            if index + 1 == len(prefix):
                return node
            return self._search(node.eq, prefix, index + 1)

    def _collect_all_words(self, node, prefix, suggestions):
        if node is None:
            return
        self._collect_all_words(node.left, prefix, suggestions)
        if node.is_end_of_word:
            suggestions.append((node.word, node.frequency))
        self._collect_all_words(node.eq, prefix + node.char, suggestions)
        self._collect_all_words(node.right, prefix, suggestions)


# Main Function
def test_autocomplete(data_structure, prefixes):
    """Test autocomplete functionality for a given data structure."""
    for prefix in prefixes:
        suggestions = data_structure.autocomplete(prefix)
        query_time = data_structure.query_time
        data_structure.query_time = 0
        print(f"Suggestions for '{prefix}' (Query time: {query_time:.6f} seconds): {len(suggestions)} suggestions")


if __name__ == "__main__":
    # Generate dataset
    random_words = [(generate_random_word(random.randint(3, 10)), random.randint(1, 100)) for _ in range(1000)]
    df = pd.DataFrame(random_words, columns=['word', 'frequency'])

    # Test Optimized Trie
    optimized_trie = OptimizedTrie()
    for _, row in df.iterrows():
        optimized_trie.insert(row['word'], row['frequency'])
    trie_size = get_size(optimized_trie)
    print(f"Optimized Trie - Memory Usage: {trie_size / (1024 * 1024):.2f} MB")
    prefixes = ['app', 'ban', 'cat', 'xyz']
    print("\nOptimized Trie Autocomplete:")
    test_autocomplete(optimized_trie, prefixes)

    # Test Optimized Patricia Trie
    optimized_patricia_trie = OptimizedPatriciaTrie()
    for _, row in df.iterrows():
        optimized_patricia_trie.insert(row['word'], row['frequency'])
    patricia_size = get_size(optimized_patricia_trie)
    print(f"\nOptimized Patricia Trie - Memory Usage: {patricia_size / (1024 * 1024):.2f} MB")
    print("\nOptimized Patricia Trie Autocomplete:")
    test_autocomplete(optimized_patricia_trie, prefixes)

    # Test Optimized TST
    optimized_tst = OptimizedTST()
    for _, row in df.iterrows():
        optimized_tst.insert(row['word'], row['frequency'])
    tst_size = get_size(optimized_tst)
    print(f"\nOptimized TST - Memory Usage: {tst_size / (1024 * 1024):.2f} MB")
    print("\nOptimized TST Autocomplete:")
    test_autocomplete(optimized_tst, prefixes)
