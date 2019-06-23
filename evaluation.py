# Python version of the evaluation script from CoNLL'00-
# Originates from: https://github.com/spyysalo/conlleval.py


# Intentional differences:
# - accept any space as delimiter by default
# - optional file argument (default STDIN)
# - option to set boundary (-b argument)
# - LaTeX output (-l argument) not supported
# - raw tags (-r argument) not supported

# add function :evaluate(predicted_label, ori_label): which will not read from file

import sys
import re
import codecs
import param
from collections import defaultdict, namedtuple

ANY_SPACE = '<SPACE>'


class FormatError(Exception):
    pass


Metrics = namedtuple('Metrics', 'tp fp fn prec rec fscore')


class EvalCounts(object):
    def __init__(self):
        self.correct_chunk = 0    # number of correctly identified chunks
        self.correct_tags = 0     # number of correct chunk tags
        self.found_correct = 0    # number of chunks in corpus
        self.found_guessed = 0    # number of identified chunks
        self.token_counter = 0    # token counter (ignores sentence breaks)

        # counts by type
        self.t_correct_chunk = defaultdict(int)
        self.t_found_correct = defaultdict(int)
        self.t_found_guessed = defaultdict(int)


def parse_tag(t: str) -> (str, str):
    m = re.match(r'^([^-]*)-(.*)$', t)
    return m.groups() if m else (t, '')


def get_tag(tid: int) -> (str, str):
    if tid < 0:
        return ('-X-', '-X-')
    if tid == 0:
        t_type = 'N'
    elif tid >> 1 << 1 == tid:
        t_type = 'I'
    else:
        t_type = 'B'

    return (t_type, param.NER_TAG[int((tid + 1)/2)])


def basic_evaluate(predicts: list, refers: list):

    counts = EvalCounts()
    num_features = None       # number of features per line
    in_correct = False        # currently processed chunks is correct until now
    last_correct = 'N'        # previous chunk tag in corpus
    last_correct_type = ''    # type of previously identified chunk tag
    last_guessed = 'N'        # previously identified chunk tag
    last_guessed_type = ''    # type of previous chunk tag in corpus

    for predict, refer in zip(predicts, refers):

        guessed, guessed_type = get_tag(predict)
        correct, correct_type = get_tag(refer)
        # first_item = predict[0]

        end_correct = end_of_chunk(last_correct, correct,
                                   last_correct_type, correct_type)
        end_guessed = end_of_chunk(last_guessed, guessed,
                                   last_guessed_type, guessed_type)
        start_correct = start_of_chunk(last_correct, correct,
                                       last_correct_type, correct_type)
        start_guessed = start_of_chunk(last_guessed, guessed,
                                       last_guessed_type, guessed_type)

        if in_correct:
            if (end_correct and end_guessed and
                    last_guessed_type == last_correct_type):
                in_correct = False
                counts.correct_chunk += 1
                counts.t_correct_chunk[last_correct_type] += 1
            elif (end_correct != end_guessed or guessed_type != correct_type):
                in_correct = False

        if start_correct and start_guessed and guessed_type == correct_type:
            in_correct = True

        if start_correct:
            counts.found_correct += 1
            counts.t_found_correct[correct_type] += 1
        if start_guessed:
            counts.found_guessed += 1
            counts.t_found_guessed[guessed_type] += 1
        if correct_type != '-X-':
            if correct == guessed and guessed_type == correct_type:
                counts.correct_tags += 1
            counts.token_counter += 1

        last_guessed = guessed
        last_correct = correct
        last_guessed_type = guessed_type
        last_correct_type = correct_type

    if in_correct:
        counts.correct_chunk += 1
        counts.t_correct_chunk[last_correct_type] += 1

    return counts


def uniq(iterable):
    seen = set()
    return [i for i in iterable if not (i in seen or seen.add(i))]


def calculate_metrics(correct, guessed, total):
    tp, fp, fn = correct, guessed - correct, total - correct
    p = 0 if tp + fp == 0 else 1.*tp / (tp + fp)
    r = 0 if tp + fn == 0 else 1.*tp / (tp + fn)
    f = 0 if p + r == 0 else 2 * p * r / (p + r)
    if p > 1:
        print(tp, fp, fn)
    return Metrics(tp, fp, fn, p, r, f)


def metrics(counts):
    c = counts
    overall = calculate_metrics(
        c.correct_chunk, c.found_guessed, c.found_correct
    )
    by_type = {}
    for t in uniq(list(c.t_found_correct) + list(c.t_found_guessed)):
        by_type[t] = calculate_metrics(
            c.t_correct_chunk[t], c.t_found_guessed[t], c.t_found_correct[t]
        )
    return overall, by_type


def report(counts, out=None):
    if out is None:
        out = sys.stdout

    overall, by_type = metrics(counts)

    c = counts
    out.write('processed %d tokens with %d phrases; ' %
              (c.token_counter, c.found_correct))
    out.write('found: %d phrases; correct: %d.\n' %
              (c.found_guessed, c.correct_chunk))

    if c.token_counter > 0:
        out.write('accuracy: %6.2f%%; ' %
                  (100.*c.correct_tags/c.token_counter))
        out.write('precision: %6.2f%%; ' % (100.*overall.prec))
        out.write('recall: %6.2f%%; ' % (100.*overall.rec))
        out.write('FB1: %6.2f\n' % (100.*overall.fscore))

    for i, m in sorted(by_type.items()):
        out.write('%17s: ' % i)
        out.write('precision: %6.2f%%; ' % (100.*m.prec))
        out.write('recall: %6.2f%%; ' % (100.*m.rec))
        out.write('FB1: %6.2f  %d\n' % (100.*m.fscore, c.t_found_guessed[i]))


def evaluate_ner(predicts, refers):
    c = basic_evaluate(predicts, refers)
    overall, by_type = metrics(c)
    p, r, f1 = 100.*overall.prec, 100.*overall.rec, 100.*overall.fscore
    result = f'{100.*c.correct_tags / c.token_counter:.2f}|'

    for ii in param.NER_TAG.values():
        if ii == '':
            continue
        if not ii in by_type:
            result += f'0.00|0.00|0.00|'
        else:
            m = by_type[ii]
            if (m.prec > 1):
                print(predicts, refers)
            result += f'{100.*m.prec:.2f}|{100.*m.rec:.2f}|{100.*m.fscore:.2f}|'
    return p, r, f1, result


def end_of_chunk(prev_tag, tag, prev_type, type_):
    # check if a chunk ended between the previous and current word
    # arguments: previous and current chunk tags, previous and current types
    chunk_end = False
    if prev_tag == 'B' and not tag == 'I':
        chunk_end = True
    if prev_tag == 'I' and not tag == 'I':
        chunk_end = True

    if prev_tag == '-X-':
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    # check if a chunk started between the previous and current word
    # arguments: previous and current chunk tags, previous and current types
    chunk_start = False

    if tag == 'B':
        chunk_start = True
    if prev_tag == 'N' and not tag == 'N':
        chunk_start = True

    return chunk_start


if __name__ == '__main__':
    predicts = [0, 0, 0, -1, 1, 2, 0, 0, -1, 1, 2, -1]
    refers = [0, 0, 0, -1, 1, 2, 0, 0, -1, 1, 4, -1]
    print(evaluate_ner(predicts, refers))
