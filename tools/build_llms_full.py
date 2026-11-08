#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regenerate llms-full.txt: README.md + docs/*.md concatenated for single-fetch reading.

The section list mirrors llms.txt; run from the repository root:

    python tools/build_llms_full.py
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SECTIONS = [
    ('README', 'README.md'),
    ('PAPER (title and abstract)', 'docs/paper.md'),
    ('HEADLINE RESULTS', 'docs/results.md'),
    ('FAQ', 'docs/faq.md'),
    ('GLOSSARY', 'docs/glossary.md'),
    ('BLOCK AND SCRIPT MAP', 'docs/blocks.md'),
    ('REGISTERED PREDICTIONS', 'docs/predictions.md'),
    ('REPRODUCTION GUIDE', 'REPRODUCE.md'),
    ('THIRD-PARTY COMPONENTS', 'THIRD_PARTY.md'),
]

HEADER = (
    "# LawKAN — full repository digest for language models\n"
    "Generated from README.md and docs/*.md; links are relative to the repository root\n"
    "(https://github.com/occamolab/lawkan).\n"
)


def main():
    parts = [HEADER]
    for title, rel in SECTIONS:
        with open(os.path.join(REPO, rel), encoding='utf-8') as f:
            body = f.read().rstrip('\n')
        parts.append(f"<!-- ===== {title} ===== -->\n\n{body}\n")
    out = '\n'.join(parts)
    with open(os.path.join(REPO, 'llms-full.txt'), 'w', encoding='utf-8') as f:
        f.write(out)
    print(f'wrote llms-full.txt ({len(out.splitlines())} lines, {len(SECTIONS)} sections)')


if __name__ == '__main__':
    main()
