"""Train the BPE tokenizer on data/tea.txt.

Usage:
    python scripts/train_tokenizer.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pythonlm.tokenizer import train_tokenizer

if __name__ == "__main__":
    train_tokenizer()
