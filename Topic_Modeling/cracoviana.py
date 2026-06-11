# -*- coding: utf-8 -*-
"""
Calculates *cracoviana* text similarity measure.
Last modification: 02.01.2025
"""

from __future__ import print_function
import sys
import os
import csv
import argparse
import logging
from collections.abc import Mapping

class DataFormatError (Exception):
    pass

class Measure:

    def __init__ (self, data_a, data_b, **kwargs):

        # default parameters

        self.frequency_threshold = 0
        self.word_limit = 0
        self.keyword_limit = 1000
        self.all_tokens = False
        self.data_type = None
        self.separator = '\t'
        self.mode = 'union'
        self.to_lower = False
        self.skip_upper_initial = False
        self.save = ''
        self.tokenize_with_nltk = True
        self.round = 4

        # kwargs parameters

        for name, val in kwargs.items ():
            setattr (self, name, val)

        self.frequency_dict_a, self.path_a = self._load_data (data_a)
        self.frequency_dict_b, self.path_b = self._load_data (data_b)
        if self.save:
            path_a = ''
            path_b = ''
            if self.save[0] in ['A', 'AB']:
                try:
                    path_a = self.save[1]
                except IndexError:
                    path_a = 'freq_a.tsv'
                if self.save[0] == 'AB':
                    try:
                        path_b = self.save[2]
                    except IndexError:
                        path_b = 'freq_b.tsv'
            elif self.save[0] == 'B':
                try:
                    path_b = self.save[1]
                except IndexError:
                        path_b = 'freq_b.tsv'
            if path_a:
                self._save_freq (self.frequency_dict_a, path_a)
            if path_b:
                self._save_freq (self.frequency_dict_b, path_b)
        self.word_distance = {}

    def _save_freq (self, dct, path):

        dct = sorted (dct.items (), key = lambda x: -x[1])
        with open (path, 'w', encoding = 'utf8') as fout:
            for word, freq in dct:
                fout.write (f'{word}\t{freq}\n')

    def _load_data (self, data):
        if type (data) == str:
            if os.path.isdir (data):
                return self._load_data_from_dir (data), data
            elif os.path.isfile (data):
                return self._load_data_from_file (data), data
            else:
                return self._load_data_from_string (data), data
            # else:
            #     error = f'"{data}" is not file nor directory or the text is too short. Aborting!'
            #     logging.error (error)
            #     raise DataFormatError (error)
        elif type (data) == list:
            freq = {}
            for word in data:
                if word in freq:
                    freq[word] += 1
                else:
                    freq[word] = 1
            return freq, 'list'
        elif isinstance (data, Mapping):
            return data, 'dict'

        raise DataFormatError (f'Wrong data type ({type (data)})')

    def _load_data_from_dir (self, path):

        dct = {}
        for fpath in [os.path.join (path, fname) for fname in os.listdir (path)]:
            cur_dct = self._load_data_from_file (fpath)
            for key, freq in cur_dct.items ():
                if key in dct:
                    dct[key] += freq
                else:
                    dct[key] = freq

        return dct
    
    def _load_data_from_file (self, path):

        logging.info (f'Loading {path}')

        with open (path, encoding = 'utf8', errors = 'surrogateescape') as fin:
            data = fin.read ()

        return self._load_data_from_string (data)

    def _load_data_from_string (self, data, data_type = None, separator = None, all_tokens = False,
                   word_limit = None, frequency_threshold = None, to_lower = False, skip_upper_initial = False):

        # if the optional parameters are not set, set them to class instance default

        def detect_data_type (data, sep = ',;\t'):

            guess = {'sep': [0] * len (sep), 'words': 0, 'count': 0}
            lines = data[:1000].splitlines ()
            for ind, line in enumerate (lines):
                if ind == 10:
                    break
                for ind, s in enumerate (sep):
                    guess['sep'][ind] += len (line.split (s))
                guess['words'] += len (line.split ())
                guess['count'] += 1
            for ind in range (len (sep)):
                if guess['sep'][ind] == 2 * guess['count']:
                    return 'csv'
            if guess['words'] == guess['count']:
                return 'word-per-line'
            else:
                return 'text'
            
        def get_tokenize_function ():
            if 'tokenize' in locals ():
                return tokenize
            tokenize = None
            if self.tokenize_with_nltk:
                try:
                    from nltk import word_tokenize as tokenize
                    try:
                        test = 'Abc. Def\nXyz'
                        tokenize (test)
                    except LookupError:
                        import nltk
                        logging.warning ('NLTK punkt_tab not found, trying to download...')
                        try:
                            nltk.download ('punkt_tab')
                            logging.warning ('OK')
                            tokenize (test)
                        except:
                            logging.warning ('NLTK resource download failed, fall back to the basic tokenizer')
                            tokenize = None
                except ImportError:
                    logging.warning ('NLTK not found, fall back to the basic tokenizer')
            if tokenize is None:
                tokenize = str.split
            return tokenize

        data_type = data_type or self.data_type
        separator = separator or self.separator
        all_tokens = all_tokens or self.all_tokens
        word_limit = word_limit or self.word_limit
        to_lower = to_lower or self.to_lower
        skip_upper_initial = skip_upper_initial or self.skip_upper_initial
        frequency_threshold = frequency_threshold or self.frequency_threshold
        dct = {}
        lines = data.splitlines ()
        if data_type is None:
            data_type = detect_data_type (data, separator)
            logging.info (f'detected data type: {data_type}')
        if data_type != 'csv':
            tokenize = get_tokenize_function ()
            for line in lines:
                if data_type == 'text':
                    if to_lower:
                        line = line.lower ()
                    tokens = tokenize (line)
                else:
                    tokens = [line]
                for token in tokens:
                    if not (all_tokens or token.isalpha ()) or (skip_upper_initial and token[0].isupper ()):
                        continue
                    if token in dct:
                        dct[token] += 1
                    else:
                        dct[token] = 1
        else:
            sniff = csv.Sniffer ()
            sample = ''.join (lines[:20])
            dialect = sniff.sniff (sample, delimiters = ',;\t')
            if sniff.has_header (sample):
                del lines[0]
            rd = csv.reader (lines, dialect)
            for ind, row in enumerate (rd):
                if word_limit and ind >= word_limit:
                    break
                try:
                    s, fr = row[:2]
                    fr = int (fr)
                except ValueError:
                    error = f'Invalid line ({ind + 1}): "{row}"'
                    logging.error (error)
                    raise DataFormatError (error)
                if not (all_tokens or s.isalpha ()) or (skip_upper_initial and token[0].isupper ()):
                    continue
                if to_lower:
                    s = s.lower ()
                    if s in dct:
                        dct[s] += fr
                    else:
                        dct[s] = fr
                else:
                    dct[s] = fr
        if frequency_threshold:
            dct = {word: freq for word, freq in dct.items () if freq >= frequency_threshold}
        logging.info ('OK')

        return dct

    def _count_unit_distances (self, sequence_a, sequence_b):
        pass

    def get_distance (self):
        pass

class Cracoviana (Measure):

    def __init__ (self, path_a, path_b, **kwargs):
        self.medal = True
        self.keywords = []
        self.keywords_df = None
        self.keyword_mode = 'AB'
        self.distance_type = 'relative'
        super ().__init__ (path_a, path_b, **kwargs)
        self.rank_dict_a = self.get_rank_dict (self.frequency_dict_a)
        self.rank_dict_b = self.get_rank_dict (self.frequency_dict_b)

    def get_rank_dict (self, frequency_dict):
        dct = {}
        freq = sorted (frequency_dict.items (), key = lambda x: -x[1])
        for ind, entry in enumerate (freq):
            dct[entry[0]] = ind

        return dct

    def _make_ziggurat (self, freq_dct):

        '''Sort a wordlist by frequency and assign relative 'position' to each element
           Example: (cat, 5), (axolotl, 2), (dog, 2), (pangolin, 1), (fox, 10) ->
                fox         0.5
                cat         0.75
                dog         0.85
                axolotl     0.95
                pangolin    1.0
            Since the two words (axolotl and dog) have the same frequency they can be ordered
            arbitrarily (like in the example above) or - when the medal parameter is set to True -
            can occupy the same position:
                fox         0.5
                cat         0.75
                dog         0.85
                axolotl     0.95
                pangolin    0.95'''

        freq_list = sorted (freq_dct.items (), key = lambda x: -x[1])

        size = float (sum ([freq for word, freq in freq_list]))
        ziggurat = []
        cumulation = 0.
        prev_gain = 0
        for word, freq in freq_list:
            gain = freq / size
            if not ziggurat:
                ziggurat.append ((word, gain))
                prev_gain = gain
                continue
            if self.medal and gain == prev_gain:
                ziggurat.append ((word, ziggurat[-1][1]))
                cumulation += gain
            else:
                ziggurat.append ((word, ziggurat[-1][1] + gain + cumulation))
                cumulation = 0.
            prev_gain = gain

        return dict (ziggurat)

    def _reverse_list (self, dct, addenda):

        for word in addenda:
            dct[word] = 1.
        words, freqs = zip (*dct.items ())
        words = reversed (words)
        return dict (zip (words, freqs))

    def _count_unit_distances (self, mode = None):

        mode = mode or self.mode

        ziggurat_a = self._make_ziggurat (self.frequency_dict_a)

        ziggurat_b = self._make_ziggurat (self.frequency_dict_b)
        words_a = set (self.frequency_dict_a.keys ())
        words_b = set (self.frequency_dict_b.keys ())
        ziggurat_rev = self._reverse_list (ziggurat_a, words_b.difference (words_a))
        if self.mode == 'union':
            words = words_a.union (words_b)
        elif self.mode == 'intersection':
            words = set (words_a).intersection (words_b)
        max_distances = []
        for word in words:
            try:
                position_a = ziggurat_a[word]
            except KeyError:
                position_a = 1.
            try:
                position_b = ziggurat_b[word]
            except KeyError:
                position_b = 1.
            self.word_distance[word] = position_b - position_a
            max_distances.append (ziggurat_rev[word] - position_a)
        self.max_distance =sum ([abs (val) for val in max_distances])

    def _find_keywords (self, keyword_mode = None, keyword_limit = None):

        keyword_mode = keyword_mode or self.keyword_mode
        keyword_limit = keyword_limit or self.keyword_limit

        self.keywords = []

        if not self.word_distance:
            self._count_unit_distances ()
        if keyword_mode.upper () == 'A':
            champions = sorted (self.word_distance.items (), key = lambda x: -x[1])
        elif keyword_mode.upper () == 'B':
            champions = sorted (self.word_distance.items (), key = lambda x: x[1])
        else:
            champions = sorted (self.word_distance.items (), key = lambda x: -abs (x[1]))
        if keyword_limit:
            champions = champions[:self.keyword_limit]
        for ch in champions:
            word, distance = ch
            rank_a = self.rank_dict_a.get (word, None)
            rank_b = self.rank_dict_b.get (word, None)
            if rank_a is not None:
                rank_a += 1
            else:
                rank_a = 0
            if rank_b is not None:
                rank_b += 1
            else:
                rank_b = 0
            self.keywords.append ((word, distance, rank_a, rank_b))

    def _keywords_as_dataframe (self):
        try:
            import pandas as pd
        except ImportError:
            logging.warning ('Trying to format keywords as a dataframe but no *pandas* module found. Please, install it first!')
            return self.keywords
        if self.keywords_df is None:
            self.keywords_df = pd.DataFrame (self.keywords, columns = ['Word', 'Distance', 'Rank A', 'Rank B'])
        return self.keywords_df

    def calculate (self):
        self._count_unit_distances ()
        self._find_keywords ()

    def get_distance (self, distance_type = None):
        distance_type = distance_type or self.distance_type
        if not self.word_distance:
            self._count_unit_distances ()
        sum_score = sum ([abs (val) for val in self.word_distance.values ()])
        average_score = sum_score / len (self.word_distance)
        if distance_type == 'sum':
            return round (sum_score, self.round)
        elif distance_type == 'average':
            return round (average_score, self.round)
        elif distance_type == 'max':
            return round (sum_score / self.max_distance, self.round)
        else:
            return round (sum_score, self.round), round (average_score, self.round)

    def get_keywords (self, keyword_mode = None, as_dataframe = False):
        if not self.keywords or keyword_mode and keyword_mode != self.keyword_mode:
            self._find_keywords (keyword_mode)
        if as_dataframe:
            return self._keywords_as_dataframe ()
        else:
            return self.keywords

    def print_results (self, path = '', mode = 'w'):

        if path:
            fout = open (path, mode, encoding = 'utf8')
            tmp_std = sys.stdout
            sys.stdout = fout
        print (f'LIST A: {self.path_a}')
        print (f'LIST B: {self.path_b}\n')
        print (f'Distance: {self.get_distance ()}\n')
        print ('*** KEYWORDS', end = ' ')
        if self.keyword_mode == 'A':
            print ('(LIST A vs LIST B) ***')
        elif self.keyword_mode == 'B':
            print ('(LIST B vs LIST A) ***')
        else:
            print ('(LIST A - positive values, LIST B - negative) ***')
        print ('-' * 70)
        for word, distance, rank_a, rank_b in self.keywords:
            if rank_a:
                rank_a += 1
            else:
                rank_a = '--'
            if rank_b:
                rank_b += 1
            else:
                rank_b = '--'
            distance = round (distance, 3)
            distance_str = f'{distance:<6}' if distance < 0 else f' {distance:<5}'
            print (f'{word:<30}{rank_a:<12}{rank_b:<12}{distance_str}')
        print ('-' * 70)
        if path:
            fout.close ()
            sys.stdout = tmp_std

class PseudoCracoviana (Measure):

    def _count_unit_distances (self):
        words_a = set (self.frequency_dict_a.keys ())
        words_b = set (self.frequency_dict_b.keys ())
        if self.mode == 'union':
            words = words_a.union (words_b)
        elif self.mode == 'intersection':
            words = set (words_a).intersection (words_b)
        N1 = float (sum (self.frequency_dict_a.values ()))
        N2 = float (sum (self.frequency_dict_b.values ()))

        for word in words:
            freq_a = self.frequency_dict_a.get (word, 0.) / N1
            freq_b = self.frequency_dict_b.get (word, 0.) / N2
            self.word_distance[word] = min (freq_a, freq_b)

    def get_distance (self):
        if not self.word_distance:
            self._count_unit_distances ()

        return round (1 - sum (self.word_distance.values ()), self.round)
    
def distance (data_a, data_b, measure = 'Cracoviana', **kwargs):
    try:
        cls = getattr (sys.modules[__name__], measure)
    except AttributeError:
        error = f'Measure {measure} is not defined'
        logging.error (error)
        raise AttributeError (error)
    instance = cls (data_a, data_b, **kwargs)
    logging.warning ('instance created')
    return instance.get_distance ()

def keywords (data_a, data_b, **kwargs):
    instance = Cracoviana (data_a, data_b, **kwargs)
    kkwargs = {}
    if 'keyword_mode' in kwargs:
        kkwargs['keyword_mode'] = kwargs['keyword_mode']
    if 'as_dataframe' in kwargs:
        kkwargs['as_dataframe'] = kwargs['as_dataframe']
    print (kkwargs)
 
    return instance.get_keywords (**kkwargs)

def get_results (data_a, data_b, measure = 'Cracoviana', **kwargs):
    try:
        cls = getattr (sys.modules[__name__], measure)
    except AttributeError:
        error = f'Measure {measure} is not defined'
        logging.error (error)
        raise AttributeError (error)
    instance = cls (data_a, data_b, **kwargs)
    instance.calculate ()
    return instance.get_distance (), instance.get_keywords (as_dataframe = kwargs.get ('keywords_as_dataframe', False))



if __name__ == '__main__':
    parser = argparse.ArgumentParser ()

    parser.add_argument ('path_a')
    parser.add_argument ('path_b')
    parser.add_argument ('-lk', '--keyword-limit', type = int, default = 100, help = 'Number of top keywords in the output. Default value: 100')
    parser.add_argument ('-lw', '--word-limit', type = int, help = 'Number of most frequent word from the frequency lists to consider. By default all the words are considered')
    parser.add_argument ('-lf', '--frequency-threshold', type = int, help = 'Frequency threshold for a word from the frequency lists to be considered. By default all the words are considered')
    parser.add_argument ('-o', '--output', help = 'Path to the output file (without this option resulta are printed to the standard output)'),
    parser.add_argument ('-O', '--output-append', help = 'Append new results to the output file instead of overwriting it')
    parser.add_argument ('-s', '--separator', help = 'Specify the separator in the csv input file if the input is a frequency list. "Tab" is default')
    parser.add_argument ('-M', '--measure', default = 'crv', choices = ['crv', 'pseudo'])
    parser.add_argument ('-m', '--mode', choices = ['union', 'intersection', 'u', 'i'],
        help = 'Method for dealing with words that are on one list only. Union (or u) adds missing words to the end of the second list, intersection (i) takes only words that are on both lists')
    parser.add_argument ('-d', '--distance-type', choices = ['average', 'sum', 'both'])
    parser.add_argument ('-a', '--all-tokens', action = 'store_true', help = 'Consider non-alpha tokens (punctuation, numbers etc.)')
    parser.add_argument ('-km', '--keyword-mode', choices = ['A', 'B', 'AB'], help = 'Show keywords only for list A, list B or for both lists (default)')
    parser.add_argument ('-v', '--verbose', action = 'store_true', help = 'Provide some info about progress')
    parser.add_argument ('-r', '--round', type = int)
    parser.add_argument ('-l', '--to-lower', action = 'store_true', help = 'convert text to lowercase')
    parser.add_argument ('-su', '--skip-upper-initial', action = 'store_true', help = 'skip words that start with upper character')
    parser.add_argument ('-S', '--save', metavar = ('A/B/AB', '[PATH]'), nargs = '+', help = 'Save calculated frequency list(s), format: A path_a or B path_b or AB path_a path_b')
    parser.add_argument ('-dt', '--data-type', choices = ['csv', 'tsv', 'text', 'word-per-line', 'wpl'], help = 'Specify input data type in case script cannot deduce it correctly')

    args = parser.parse_args ()
    if args.separator in ['t', 'tab']:
        args.separator = '\t'
    if args.output:
        args.output_mode = 'w'
    if args.output_append:
        args.output_mode = 'a'
        args.output = args.output_append
    if args.data_type == 'tsv':
        args.data_type = 'csv'
        args.separator = '\t'
    if args.data_type == 'wpl':
        args.data_type = 'word-per-line'
    if args.mode == 'i':
        args.mode = 'intersection'
    if args.mode == 'u':
        args.mode = 'union'
    if not args.save:
        args.save = ''
    logging.basicConfig (format = '[%(levelname)s] %(message)s')
    if args.verbose:
        logger = logging.getLogger ()
        logger.setLevel (logging.INFO)

        logging.info ('Verbose mode')

    if args.measure == 'crv':
        measure_arguments = ['mode', 'separator', 'keyword_mode', 'keyword_limit', 'word_limit', 'frequency_threshold', 'distance_type', 'data_type', 'all_tokens', 'save']
        crv_args = {}
        for argument in measure_arguments:
            value = getattr (args, argument)
            if value != None:
                crv_args[argument] = value
        crv = Cracoviana (args.path_a, args.path_b, **crv_args)
        crv.calculate ()
        crv.print_results (args.output, args.output_mode if args.output else '')

    elif args.measure == 'pseudo':
        measure_arguments = ['separator', 'word_limit', 'frequency_threshold', 'data_type', 'all_tokens', 'save']
        ps_args = {}
        for argument in measure_arguments:
            value = getattr (args, argument)
            if value:
                ps_args[argument] = value
        pseudo = PseudoCracoviana (args.path_a, args.path_b, **ps_args)
        print (pseudo.get_distance ())
