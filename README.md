# Autocomplete System Using Trie Variants

## Overview
This project implements an **Autocomplete System** using different data structures: **Standard Trie**, **Patricia Trie**, and **Ternary Search Tree (TST)**. The system supports efficient word insertion and prefix-based autocomplete functionality, making it ideal for applications requiring text prediction and search.

## Features
- **Standard Trie**:
  - Classical trie structure for string storage and prefix queries.
- **Patricia Trie**:
  - Optimized for storage by compressing edges with common prefixes.
- **Ternary Search Tree (TST)**:
  - Hybrid approach combining binary search and trie for compact memory usage.
- **Autocomplete**:
  - Suggests words based on prefix input, sorted by frequency.
- **Performance Metrics**:
  - Measures insertion time, query latency, and memory usage for each structure.

## Key Components
- **Insertion**:
  - Words and their associated frequencies are stored in the data structures.
- **Autocomplete**:
  - Prefix-based retrieval of suggestions sorted by frequency.
- **Performance Evaluation**:
  - Comparison of speed and memory usage across the three structures.

## Implementation Details
- **Programming Language**: Python
- **Data Structures**:
  - **Standard Trie**: Recursive node-based implementation.
  - **Patricia Trie**: Path compression for efficient memory usage.
  - **TST**: Binary search for character nodes, supporting compact storage.
- **Sample Dataset**:
  - A collection of words with their usage frequencies.


