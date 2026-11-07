# Python Programming Guide

## Overview
Python is a high-level, interpreted programming language known for its simplicity and readability.
It was created by Guido van Rossum and first released in 1991.

## Key Features
- **Dynamic typing**: Variables don't need explicit type declarations
- **Indentation-based syntax**: Uses whitespace to define code blocks
- **Extensive standard library**: "Batteries included" philosophy
- **Multiple paradigms**: Supports OOP, functional, and procedural programming

## Data Types
Python has several built-in data types:
1. **int**: Integer numbers (e.g., 42, -7, 0)
2. **float**: Floating-point numbers (e.g., 3.14, -0.001)
3. **str**: Strings (e.g., "hello", 'world')
4. **list**: Ordered mutable sequences (e.g., [1, 2, 3])
5. **dict**: Key-value mappings (e.g., {"key": "value"})

## List Comprehensions
List comprehensions provide a concise way to create lists:
```python
squares = [x**2 for x in range(10)]
even_squares = [x**2 for x in range(10) if x % 2 == 0]
```

## Virtual Environments
Always use virtual environments for Python projects:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```
