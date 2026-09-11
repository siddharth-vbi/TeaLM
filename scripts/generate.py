"""Generate text from a trained PythonLM checkpoint.

Usage:
    python scripts/generate.py
    python scripts/generate.py --prompt "How to make green tea" --tokens 100
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
from pythonlm import config
from pythonlm.model import PythonLM
from pythonlm.tokenizer import load_tokenizer, decode


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text with PythonLM")
    parser.add_argument("--prompt",      type=str,   default="To make tea")
    parser.add_argument("--tokens",      type=int,   default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k",       type=int,   default=20)
    parser.add_argument("--checkpoint",  type=str,   default=config.BEST_CHECKPOINT)
    return parser.parse_args()


def main():
    args = parse_args()

    device    = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer()

    model = PythonLM(vocab_size=tokenizer.get_vocab_size()).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    encoded = tokenizer.encode(args.prompt).ids
    idx     = torch.tensor([encoded], dtype=torch.long, device=device)

    output = model.generate(
        idx,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    text = decode(tokenizer, output[0].tolist())
    print(f"\nPrompt    : {args.prompt}")
    print(f"Generated :\n{text}")


if __name__ == "__main__":
    main()
