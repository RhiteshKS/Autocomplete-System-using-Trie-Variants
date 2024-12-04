"""
Standard Trie Variants Implementation
-------------------------------------
This script includes:
1. Standard Trie
2. Patricia Trie
3. Ternary Search Tree (TST)

Features:
- Insert words with frequency.
- Autocomplete functionality.
- Performance metrics: insertion time, query time, and memory usage.

Author: [Rhitesh Kumar Singh]
"""

import pandas as pd
import time
import sys
import random
import string


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


class TrieNode:
    """Node for the Standard Trie."""
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.frequency = 0
        self.word = None


class Trie:
    """Standard Trie implementation."""
    def __init__(self):
        self.root = TrieNode()
        self.insert_time = 0
        self.query_time = 0

    def insert(self, word, frequency):
        start_time = time.time()
        node = self.root
        for char in word:
            node = node.children.setdefault(char, TrieNode())
        node.is_end_of_word = True
        node.frequency = frequency
        node.word = word
        self.insert_time += time.time() - start_time

    def autocomplete(self, prefix):
        start_time = time.time()
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                self.query_time += time.time() - start_time
                return []
        suggestions = []
        self._dfs(node, suggestions)
        self.query_time += time.time() - start_time
        return sorted(suggestions, key=lambda x: x[1], reverse=True)

    def _dfs(self, node, suggestions):
        if node.is_end_of_word:
            suggestions.append((node.word, node.frequency))
        for child in node.children.values():
            self._dfs(child, suggestions)


class PatriciaTrieNode:
    """Node for the Patricia Trie."""
    def __init__(self, key):
        self.key = key
        self.children = {}
        self.is_end_of_word = False
        self.frequency = 0
        self.word = None


class PatriciaTrie:
    """Patricia Trie implementation."""
    def __init__(self):
        self.root = PatriciaTrieNode("")
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
            for child in node.children.values():
                common_prefix_len = self._common_prefix_length(word[i:], child.key)
                if common_prefix_len > 0:
                    if common_prefix_len == len(child.key):
                        node = child
                        i += common_prefix_len
                    else:
                        # Split the node
                        existing_child = PatriciaTrieNode(child.key[common_prefix_len:])
                        existing_child.children = child.children
                        existing_child.is_end_of_word = child.is_end_of_word
                        existing_child.frequency = child.frequency
                        existing_child.word = child.word

                        child.key = child.key[:common_prefix_len]
                        child.children = {existing_child.key[0]: existing_child}
                        child.is_end_of_word = False
                        child.frequency = 0
                        child.word = None

                        node = child
                        i += common_prefix_len
                    found = True
                    break
            if not found:
                new_node = PatriciaTrieNode(word[i:])
                new_node.is_end_of_word = True
                new_node.frequency = frequency
                new_node.word = word
                node.children[new_node.key[0]] = new_node
                break
        else:
            node.is_end_of_word = True
            node.frequency = frequency
            node.word = word
        self.insert_time += time.time() - start_time

    def autocomplete(self, prefix):
        start_time = time.time()
        node = self.root
        i = 0
        while i < len(prefix):
            found = False
            for child in node.children.values():
                common_prefix_len = self._common_prefix_length(prefix[i:], child.key)
                if common_prefix_len > 0:
                    if common_prefix_len == len(child.key):
                        node = child
                        i += common_prefix_len
                    else:
                        self.query_time += time.time() - start_time
                        return []
                    found = True
                    break
            if not found:
                self.query_time += time.time() - start_time
                return []
        suggestions = []
        self._dfs(node, suggestions)
        self.query_time += time.time() - start_time
        return sorted(suggestions, key=lambda x: x[1], reverse=True)

    def _dfs(self, node, suggestions):
        if node.is_end_of_word:
            suggestions.append((node.word, node.frequency))
        for child in node.children.values():
            self._dfs(child, suggestions)


class TSTNode:
    """Node for the Ternary Search Tree."""
    def __init__(self, char):
        self.char = char
        self.left = None
        self.eq = None
        self.right = None
        self.is_end_of_word = False
        self.frequency = 0
        self.word = None


class TST:
    """Ternary Search Tree implementation."""
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
            node = TSTNode(char)
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
            self._collect(node.eq, prefix, suggestions)
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

    def _collect(self, node, prefix, suggestions):
        if node is None:
            return
        self._collect(node.left, prefix, suggestions)
        if node.is_end_of_word:
            suggestions.append((node.word, node.frequency))
        self._collect(node.eq, prefix + node.char, suggestions)
        self._collect(node.right, prefix, suggestions)


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

    # Test Standard Trie
    standard_trie = Trie()
    for _, row in df.iterrows():
        standard_trie.insert(row['word'], row['frequency'])
    trie_size = get_size(standard_trie)
    print(f"Standard Trie - Memory Usage: {trie_size / (1024 * 1024):.2f} MB")
    prefixes = ['app', 'ban', 'cat', 'xyz']
    print("\nStandard Trie Autocomplete:")
    test_autocomplete(standard_trie, prefixes)

    # Test Patricia Trie
    patricia_trie = PatriciaTrie()
    for _, row in df.iterrows():
        patricia_trie.insert(row['word'], row['frequency'])
    patricia_size = get_size(patricia_trie)
    print(f"\nPatricia Trie - Memory Usage: {patricia_size / (1024 * 1024):.2f} MB")
    print("\nPatricia Trie Autocomplete:")
    test_autocomplete(patricia_trie, prefixes)

    # Test TST
    tst = TST()
    for _, row in df.iterrows():
        tst.insert(row['word'], row['frequency'])
    tst_size = get_size(tst)
    print(f"\nTST - Memory Usage: {tst_size / (1024 * 1024):.2f} MB")
    print("\nTST Autocomplete:")
    test_autocomplete(tst, prefixes)
