# Autocomplete System Using Trie Variants

This repository contains **Trie data structures** (Standard and Optimized) for building an efficient **autocomplete system**. The implementation includes **Standard Tries**, **Patricia Tries**, and **Ternary Search Trees (TST)**, with **optimizations** focused on improving **memory usage** and **performance**.

## Features

- **Standard Trie**: A simple, well-known Trie structure that efficiently supports **word insertions** and **autocomplete** queries.
- **Patricia Trie**: A **compressed version of the Trie** that reduces space by merging nodes with common prefixes.
- **Ternary Search Tree (TST)**: A **space-efficient variant** combining aspects of both Tries and binary search trees for autocomplete queries.

### Optimizations:
- **Memory Pooling**: Reduces memory fragmentation by reusing Trie nodes.
- **Efficient Node Storage**: Optimized memory usage by using `__slots__` for nodes, eliminating unnecessary overhead.
- **Common-Prefix Compression** (for Patricia Trie): Minimizes the number of nodes by merging common prefixes, improving space efficiency.
- **Iterative DFS**: Avoids recursion depth limits by using iterative DFS for autocomplete suggestions.
- **Index-Based Child Access**: Optimized child node access using fixed-size arrays (26 slots for the alphabet) rather than dictionaries.
- **Performance Monitoring**: Metrics for **insertion time**, **query time**, and **memory usage**.

## Data Structures

### 1. Standard Trie
- Implements the basic **Trie structure** for efficient **prefix searching**.
- Supports **autocomplete** by recursively traversing nodes based on prefix matches.
- **Performance**: Memory usage can be higher due to redundant nodes and pointers.

### 2. Patricia Trie
- **Compressed Trie** where edges represent substrings (common prefixes) instead of individual characters.
- Reduces space complexity by merging nodes with shared prefixes.
- **Performance**: Faster than Standard Trie for large datasets due to fewer nodes and optimized storage.

### 3. Ternary Search Tree (TST)
- Combines features of **Tries** and **Binary Search Trees** (BST).
- Nodes contain a character, with **left**, **middle**, and **right** pointers for efficient searching.
- **Performance**: A balanced alternative that offers good memory and time performance, especially for non-uniform datasets.

## Optimizations in Detail

### 1. **Memory Pooling**
   - Nodes are reused from a pre-allocated pool rather than being newly created with every insertion or query.
   - Reduces the overhead of frequent memory allocation and deallocation.

### 2. **Efficient Node Storage**
   - Used `__slots__` in all node classes to avoid the creation of the `__dict__` attribute, reducing memory overhead.
   - Each node only stores the required attributes for efficient space utilization.

### 3. **Batch Insertion**
   - Insertions are timed and processed in batches to improve the overall performance and enable more accurate benchmarking.
   
### 4. **Efficient Search Logic**
   - **Patricia Trie** utilizes **common-prefix compression**, merging nodes with identical prefixes, significantly reducing the number of nodes and improving performance.
   - **Ternary Search Tree (TST)** is structured to balance between a **binary search tree** and **trie**, making it a great choice for string matching and autocomplete functionality.

### 5. **Iterative DFS for Autocomplete**
   - **Depth-First Search (DFS)** for autocomplete suggestions is done iteratively to avoid hitting recursion limits in Python and reducing the risk of stack overflow for large datasets.

### 6. **Improved Memory Efficiency**
   - For **TST** and **Patricia Trie**, node comparisons and insertions are optimized to store only relevant data and minimize node creation overhead.
   - **Index-based child node access** using fixed-size arrays instead of dynamic structures like dictionaries reduces memory consumption.

### 7. **Query Time and Insertion Time Tracking**
   - The system tracks **query time** and **insertion time** for each Trie variant, allowing you to compare their performance in terms of time efficiency.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/trie-autocomplete.git
   cd trie-autocomplete


