# Copyright 2024 <org> Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fits a previous self-improve round's code (or plan text) into a token budget without
producing syntactically broken code. Used when substituting {previous_plan_code} into
the next self-improve prompt.

Strategy, cheapest first:
  1. Strip comment-only and blank lines.
  2. If still over budget, parse with ast and drop whole top-level statements, keeping
     imports, statements with hyperparameter assignments, and the last few statements,
     then greedily backfill the remaining budget with other statements in source order.
  3. If the text is not parseable Python, truncate to the last `budget` tokens.
"""
import ast
import io
import re
import tokenize

_HYPERPARAM_RE = re.compile(
    r'\b(max_iter|n_estimators|max_features|max_depth|n_jobs|cv|alpha|C|learning_rate'
    r'|epochs|num_epochs|batch_size|num_boost_round|n_neighbors|min_samples_split'
    r'|min_samples_leaf|ngram_range|num_leaves|subsample|colsample_bytree|reg_alpha'
    r'|reg_lambda|tol|hidden_size|num_layers|dropout|lr)\s*='
)


def _strip_comments_and_blank_lines(code: str) -> str:
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenizeError, IndentationError, SyntaxError, ValueError):
        return code
    comment_lines = {t.start[0] for t in toks if t.type == tokenize.COMMENT}
    lines = code.splitlines()
    kept = [line for i, line in enumerate(lines, start=1) if i not in comment_lines and line.strip() != '']
    return '\n'.join(kept)


def _top_level_statement_blocks(code: str):
    """
    (start_line, end_line, text) per top-level statement, in source order.
    Returns None if code doesn't parse.
    """
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return None
    lines = code.splitlines()
    blocks = []
    for node in tree.body:
        start = node.lineno
        end = getattr(node, 'end_lineno', node.lineno)
        blocks.append((start, end, '\n'.join(lines[start - 1:end])))
    return blocks


def _tail_truncate_tokens(text: str, tokenizer, budget: int) -> str:
    if budget <= 0:
        return ''
    ids = tokenizer(text)['input_ids']
    return tokenizer.decode(ids[-budget:], skip_special_tokens=True)


def truncate_code_to_budget(code: str, tokenizer, budget: int, tail_keep: int = 3) -> str:
    """Fit `code` inside `budget` tokens (measured by `tokenizer`)."""
    if not code:
        return code

    def n_tokens(s):
        return len(tokenizer(s)['input_ids'])

    stripped = _strip_comments_and_blank_lines(code)
    if n_tokens(stripped) <= budget:
        return stripped

    blocks = _top_level_statement_blocks(stripped)
    if not blocks:
        return _tail_truncate_tokens(stripped, tokenizer, budget)

    n = len(blocks)
    must_keep = [False] * n
    for i, (_, _, text) in enumerate(blocks):
        first_line = text.lstrip().split('\n', 1)[0]
        if first_line.startswith(('import ', 'from ')) or _HYPERPARAM_RE.search(text):
            must_keep[i] = True
    for i in range(max(0, n - tail_keep), n):
        must_keep[i] = True

    def render(keep_flags):
        pieces = []
        omitted_lines = 0
        for i, (start, end, text) in enumerate(blocks):
            if keep_flags[i]:
                if omitted_lines:
                    pieces.append(f'# ... {omitted_lines} line(s) omitted ...')
                    omitted_lines = 0
                pieces.append(text)
            else:
                omitted_lines += end - start + 1
        if omitted_lines:
            pieces.append(f'# ... {omitted_lines} line(s) omitted ...')
        return '\n'.join(pieces)

    current = list(must_keep)
    if n_tokens(render(current)) > budget:
        # even imports + hyperparameter lines + tail don't fit -- last resort
        return _tail_truncate_tokens(stripped, tokenizer, budget)

    # greedily backfill the optional (non-must-keep) blocks, in their
    # original source order, keeping whichever still fit the budget
    for i in range(n):
        if current[i]:
            continue
        trial = list(current)
        trial[i] = True
        if n_tokens(render(trial)) <= budget:
            current = trial

    return render(current)
