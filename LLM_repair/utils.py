import ast
import re
import sys

import pandas as pd
import numpy as np
from PIL import Image

def extract_dict(string_data):
    try:
        # Search for the simplest dictionary pattern in the string
        pattern = r'\{[^{}]*\}'
        matches = re.findall(pattern, string_data)
        if matches:
            # Iterate over all matches to find the first valid dictionary
            for match in matches:
                try:
                    mission_dict = ast.literal_eval(match)
                    return mission_dict
                except:
                    continue
            print("No valid dictionary found in the matches.")
            return {}
        else:
            print("No dictionary-like pattern found in the string.")
            return {}
    except ValueError as e:
        print(f"Error converting string to dictionary: {e}")
        return {}
    except SyntaxError as e:
        print(f"Syntax error in the string: {e}")
        return {}
    
def extract_list(string_list):
    string_list = "[" + string_list.replace("```python\n", "").replace("\n```", "").split("[")[1]
    return ast.literal_eval(string_list)

def extract_between_ticks(text):
    """
    Extracts text between the first two sets of triple backticks (```) in the given text.
    Raises an error if the triple backticks are not found or if the text between them is missing.
    """
    text.replace("'''", "```")
    # Split the text by triple backticks
    parts = text.split("```")
    
    # If there are at least three parts (beginning, desired text, end), return the desired text
    if len(parts) >= 3 and parts[1].strip():
        return parts[1].strip()
    else:
        raise ValueError("Triple backticks (```) not found or text between them is missing.")
    
def remove_extra_special_chars(input_string):
    """
    This function removes all special characters from the string except one instance of each.
    It does not consider alphabets and numbers as special characters.
    """
    # Finding all special characters (non-alphanumeric and non-whitespace)
    special_chars = set(char for char in input_string if not char.isalnum() and not char.isspace())

    # Create a dictionary to keep track of the first occurrence of each special character
    first_occurrences = {char: False for char in special_chars}

    # Process the string, keeping the first occurrence of each special character
    new_string = []
    for char in input_string:
        if char in special_chars:
            if not first_occurrences[char]:
                new_string.append(char)
                first_occurrences[char] = True
        else:
            new_string.append(char)

    return ''.join(new_string)

def extract_present_elements(grid_string, elements_dict):
    # Initialize an empty dictionary to hold the elements present in the grid
    present_elements = {}
    
    # Iterate over the dictionary items
    for key, value in elements_dict.items():
        # Check if the element's symbol is present in the grid string
        if value in grid_string:
            # If present, add the key to the present_elements dictionary
            present_elements[key] = value
            
    return present_elements

class _Sentinel:
    def __repr__(self):
        return "<implicit>"

_sentinel = _Sentinel()

def _parse_value_tb(exc, value, tb):
    if (value is _sentinel) != (tb is _sentinel):
        raise ValueError("Both or neither of value and tb must be given")
    if value is tb is _sentinel:
        if exc is not None:
            if isinstance(exc, BaseException):
                return exc, exc.__traceback__

            raise TypeError(f'Exception expected for value, '
                            f'{type(exc).__name__} found')
        else:
            return None, None
    return value, tb

# def format_exception(exc, /, value=_sentinel, tb=_sentinel, limit=None, \
#                      chain=True):
#     """Format a stack trace and the exception information.

#     The arguments have the same meaning as the corresponding arguments
#     to print_exception().  The return value is a list of strings, each
#     ending in a newline and some containing internal newlines.  When
#     these lines are concatenated and printed, exactly the same text is
#     printed as does print_exception().
#     """
#     value, tb = _parse_value_tb(exc, value, tb)
#     te = TracebackException(type(value), value, tb, limit=limit, compact=True)
#     return list(te.format(chain=chain))

# def format_exc(limit=None, chain=True):
#     """Like print_exc() but return a string."""
#     return "".join(format_exception(*sys.exc_info(), limit=limit, chain=chain))

def euclidean_distance(a, b):
    # Convert strings to lists of ASCII values
    ascii_a = np.array([ord(char) for char in a])
    ascii_b = np.array([ord(char) for char in b])
    
    # If lengths differ, truncate the longer one to match the shorter one
    min_len = min(len(ascii_a), len(ascii_b))
    ascii_a = ascii_a[:min_len]
    ascii_b = ascii_b[:min_len]
    
    # Calculate Euclidean distance
    distance = np.sqrt(np.sum((ascii_a - ascii_b) ** 2))
    
    return distance

def pad_rows_to_max_length(text):
    """
    Pads each row in the provided text to make them of the length of the row with maximum length.
    Rows are padded with the last character found in that row.
    """
    # Split the text into lines
    lines = text.strip().split("\n")
    
    # Determine the maximum line length
    max_length = max(len(line) for line in lines)
    
    # Pad each line to the maximum length
    padded_lines = [line + line[-1] * (max_length - len(line)) if line else "" for line in lines]
    
    return "\n".join(padded_lines)

def remove_extra_special_chars(input_string):
    """
    This function removes all special characters from the string except one instance of each.
    It does not consider alphabets and numbers as special characters.
    """
    # Finding all special characters (non-alphanumeric and non-whitespace)
    special_chars = set(char for char in input_string if not char.isalnum() and not char.isspace())

    # Create a dictionary to keep track of the first occurrence of each special character
    first_occurrences = {char: False for char in special_chars}

    # Process the string, keeping the first occurrence of each special character
    new_string = []
    for char in input_string:
        if char in special_chars:
            if not first_occurrences[char]:
                new_string.append(char)
                first_occurrences[char] = True
        else:
            new_string.append(char)

    return ''.join(new_string)

def parse_grid(input_str):
    grid = [list(line) for line in input_str.strip().split('\n')]
    return grid

def bert_batch_similarity(descs1, descs2):
    # """
    # Calculate similarities for batches of descriptions using DistilBERT embeddings and cosine similarity.

    # Parameters:
    # descs1 (List[str]): First list of descriptions.
    # descs2 (List[str]): Second list of descriptions.

    # Returns:
    # List[float]: List of cosine similarity scores.
    # """

    # #tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    # #model = AutoModel.from_pretrained('bert-base-uncased')

    # tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    # model = DistilBertModel.from_pretrained('distilbert-base-uncased')

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model.to(device)

    # # Tokenize and encode the batches of descriptions
    # tokens1 = tokenizer(descs1, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
    # tokens2 = tokenizer(descs2, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)

    # with torch.no_grad():
    #     embedding1 = model(**tokens1).last_hidden_state.mean(dim=1)
    #     embedding2 = model(**tokens2).last_hidden_state.mean(dim=1)

    # # Compute cosine similarities
    # similarities = [1 - cosine(e1.cpu().numpy(), e2.cpu().numpy()) for e1, e2 in zip(embedding1, embedding2)]

    # return similarities
    return 0

def find_most_similar_images(dictionary, csv_path):
    """
    Find the most semantically similar image for each tile in the dictionary.

    Parameters:
    dictionary (Dict[str, str]): A dictionary with descriptions as keys and tiles as values.
    csv_path (str): Path to the CSV file containing image filenames and descriptions.

    Returns:
    Tuple[Dict[str, Image.Image], Dict[str, int]]: A tuple containing two dictionaries,
        one with the tiles and corresponding Image objects, and the other with tiles and similarity scores.
    """
    # Load the CSV file
    folder = f'{csv_path}/world_tileset_data'
    folder_char = f'{csv_path}/character_sprite_data'
    data = pd.read_csv(f"{folder}/metadata.csv")
    data_char = pd.read_csv(f"{folder_char}/metadata.csv")
    # Initialize dictionaries for image objects and similarity scores
    images = {}
    similarity_scores = {}

    # Iterate over the dictionary to find the most similar image for each tile
    for desc, tile in dictionary.items():
        if not tile.isalpha():
            max_similarity = 0
            selected_image = None

            # Compare each description with the descriptions in the CSV
            #for _, row in data.iterrows():
            #    similarity = bert_similarity(desc, row['description'])
            #    if similarity >= max_similarity:
            #        max_similarity = similarity
            #        selected_image = row['filename']

            similarities = bert_batch_similarity([desc] * len(list(data_char['description'])), list(data_char['description']))
            max_similarity = max(similarities)
            max_index = similarities.index(max_similarity)
            selected_image = data_char.iloc[max_index]['filename']

            # Load the image and store it along with the similarity score
            if selected_image:
                #image_path = os.path.join(os.path.dirname(csv_path), selected_image)
                image_path = f"{folder_char}/{selected_image}"
                images[tile] = Image.open(image_path).convert("RGBA")
                similarity_scores[tile] = max_similarity
        else:
            max_similarity = 0
            selected_image = None

            # Compare each description with the descriptions in the CSV
            similarities = bert_batch_similarity([desc] * len(list(data['description'])), list(data['description']))
            max_similarity = max(similarities)
            max_index = similarities.index(max_similarity)
            selected_image = data.iloc[max_index]['filename']

            # Load the image and store it along with the similarity score
            if selected_image:
                #image_path = os.path.join(os.path.dirname(csv_path), selected_image)
                image_path = f"{folder}/{selected_image}"
                images[tile] = Image.open(image_path).convert("RGBA")
                similarity_scores[tile] = max_similarity

    return images, similarity_scores


def find_character_position(game_str, character):
    # Split the game_str into lines
    lines = game_str.split('\n')
    
    # Search for the character in each line
    for x, line in enumerate(lines):
        if character in line:
            y = line.index(character)
            return (x, y)  # Return as soon as the character is found

    return None  # Return None if the character is not found

def list_of_lists_to_string(lists):
    return '\n'.join([''.join(sublist) for sublist in lists])

# class TracebackException:
#     """An exception ready for rendering.

#     The traceback module captures enough attributes from the original exception
#     to this intermediary form to ensure that no references are held, while
#     still being able to fully print or format it.

#     max_group_width and max_group_depth control the formatting of exception
#     groups. The depth refers to the nesting level of the group, and the width
#     refers to the size of a single exception group's exceptions array. The
#     formatted output is truncated when either limit is exceeded.

#     Use `from_exception` to create TracebackException instances from exception
#     objects, or the constructor to create TracebackException instances from
#     individual components.

#     - :attr:`__cause__` A TracebackException of the original *__cause__*.
#     - :attr:`__context__` A TracebackException of the original *__context__*.
#     - :attr:`__suppress_context__` The *__suppress_context__* value from the
#       original exception.
#     - :attr:`stack` A `StackSummary` representing the traceback.
#     - :attr:`exc_type` The class of the original traceback.
#     - :attr:`filename` For syntax errors - the filename where the error
#       occurred.
#     - :attr:`lineno` For syntax errors - the linenumber where the error
#       occurred.
#     - :attr:`end_lineno` For syntax errors - the end linenumber where the error
#       occurred. Can be `None` if not present.
#     - :attr:`text` For syntax errors - the text where the error
#       occurred.
#     - :attr:`offset` For syntax errors - the offset into the text where the
#       error occurred.
#     - :attr:`end_offset` For syntax errors - the offset into the text where the
#       error occurred. Can be `None` if not present.
#     - :attr:`msg` For syntax errors - the compiler error message.
#     """

#     def __init__(self, exc_type, exc_value, exc_traceback, *, limit=None,
#             lookup_lines=True, capture_locals=False, compact=False,
#             max_group_width=15, max_group_depth=10, _seen=None):
#         # NB: we need to accept exc_traceback, exc_value, exc_traceback to
#         # permit backwards compat with the existing API, otherwise we
#         # need stub thunk objects just to glue it together.
#         # Handle loops in __cause__ or __context__.
#         is_recursive_call = _seen is not None
#         if _seen is None:
#             _seen = set()
#         _seen.add(id(exc_value))

#         self.max_group_width = max_group_width
#         self.max_group_depth = max_group_depth

#         self.stack = StackSummary._extract_from_extended_frame_gen(
#             _walk_tb_with_full_positions(exc_traceback),
#             limit=limit, lookup_lines=lookup_lines,
#             capture_locals=capture_locals)
#         self.exc_type = exc_type
#         # Capture now to permit freeing resources: only complication is in the
#         # unofficial API _format_final_exc_line
#         self._str = _safe_string(exc_value, 'exception')
#         self.__notes__ = getattr(exc_value, '__notes__', None)

#         if exc_type and issubclass(exc_type, SyntaxError):
#             # Handle SyntaxError's specially
#             self.filename = exc_value.filename
#             lno = exc_value.lineno
#             self.lineno = str(lno) if lno is not None else None
#             end_lno = exc_value.end_lineno
#             self.end_lineno = str(end_lno) if end_lno is not None else None
#             self.text = exc_value.text
#             self.offset = exc_value.offset
#             self.end_offset = exc_value.end_offset
#             self.msg = exc_value.msg
#         if lookup_lines:
#             self._load_lines()
#         self.__suppress_context__ = \
#             exc_value.__suppress_context__ if exc_value is not None else False

#         # Convert __cause__ and __context__ to `TracebackExceptions`s, use a
#         # queue to avoid recursion (only the top-level call gets _seen == None)
#         if not is_recursive_call:
#             queue = [(self, exc_value)]
#             while queue:
#                 te, e = queue.pop()
#                 if (e and e.__cause__ is not None
#                     and id(e.__cause__) not in _seen):
#                     cause = TracebackException(
#                         type(e.__cause__),
#                         e.__cause__,
#                         e.__cause__.__traceback__,
#                         limit=limit,
#                         lookup_lines=lookup_lines,
#                         capture_locals=capture_locals,
#                         max_group_width=max_group_width,
#                         max_group_depth=max_group_depth,
#                         _seen=_seen)
#                 else:
#                     cause = None

#                 if compact:
#                     need_context = (cause is None and
#                                     e is not None and
#                                     not e.__suppress_context__)
#                 else:
#                     need_context = True
#                 if (e and e.__context__ is not None
#                     and need_context and id(e.__context__) not in _seen):
#                     context = TracebackException(
#                         type(e.__context__),
#                         e.__context__,
#                         e.__context__.__traceback__,
#                         limit=limit,
#                         lookup_lines=lookup_lines,
#                         capture_locals=capture_locals,
#                         max_group_width=max_group_width,
#                         max_group_depth=max_group_depth,
#                         _seen=_seen)
#                 else:
#                     context = None

#                 if e and isinstance(e, BaseExceptionGroup):
#                     exceptions = []
#                     for exc in e.exceptions:
#                         texc = TracebackException(
#                             type(exc),
#                             exc,
#                             exc.__traceback__,
#                             limit=limit,
#                             lookup_lines=lookup_lines,
#                             capture_locals=capture_locals,
#                             max_group_width=max_group_width,
#                             max_group_depth=max_group_depth,
#                             _seen=_seen)
#                         exceptions.append(texc)
#                 else:
#                     exceptions = None

#                 te.__cause__ = cause
#                 te.__context__ = context
#                 te.exceptions = exceptions
#                 if cause:
#                     queue.append((te.__cause__, e.__cause__))
#                 if context:
#                     queue.append((te.__context__, e.__context__))
#                 if exceptions:
#                     queue.extend(zip(te.exceptions, e.exceptions))

#     @classmethod
#     def from_exception(cls, exc, *args, **kwargs):
#         """Create a TracebackException from an exception."""
#         return cls(type(exc), exc, exc.__traceback__, *args, **kwargs)

#     def _load_lines(self):
#         """Private API. force all lines in the stack to be loaded."""
#         for frame in self.stack:
#             frame.line

#     def __eq__(self, other):
#         if isinstance(other, TracebackException):
#             return self.__dict__ == other.__dict__
#         return NotImplemented

#     def __str__(self):
#         return self._str

#     def format_exception_only(self):
#         """Format the exception part of the traceback.

#         The return value is a generator of strings, each ending in a newline.

#         Normally, the generator emits a single string; however, for
#         SyntaxError exceptions, it emits several lines that (when
#         printed) display detailed information about where the syntax
#         error occurred.

#         The message indicating which exception occurred is always the last
#         string in the output.
#         """
#         if self.exc_type is None:
#             yield _format_final_exc_line(None, self._str)
#             return

#         stype = self.exc_type.__qualname__
#         smod = self.exc_type.__module__
#         if smod not in ("__main__", "builtins"):
#             if not isinstance(smod, str):
#                 smod = "<unknown>"
#             stype = smod + '.' + stype

#         if not issubclass(self.exc_type, SyntaxError):
#             yield _format_final_exc_line(stype, self._str)
#         else:
#             yield from self._format_syntax_error(stype)
#         if isinstance(self.__notes__, collections.abc.Sequence):
#             for note in self.__notes__:
#                 note = _safe_string(note, 'note')
#                 yield from [l + '\n' for l in note.split('\n')]
#         elif self.__notes__ is not None:
#             yield _safe_string(self.__notes__, '__notes__', func=repr)

#     def _format_syntax_error(self, stype):
#         """Format SyntaxError exceptions (internal helper)."""
#         # Show exactly where the problem was found.
#         filename_suffix = ''
#         if self.lineno is not None:
#             yield '  File "{}", line {}\n'.format(
#                 self.filename or "<string>", self.lineno)
#         elif self.filename is not None:
#             filename_suffix = ' ({})'.format(self.filename)

#         text = self.text
#         if text is not None:
#             # text  = "   foo\n"
#             # rtext = "   foo"
#             # ltext =    "foo"
#             rtext = text.rstrip('\n')
#             ltext = rtext.lstrip(' \n\f')
#             spaces = len(rtext) - len(ltext)
#             yield '    {}\n'.format(ltext)

#             if self.offset is not None:
#                 offset = self.offset
#                 end_offset = self.end_offset if self.end_offset not in {None, 0} else offset
#                 if offset == end_offset or end_offset == -1:
#                     end_offset = offset + 1

#                 # Convert 1-based column offset to 0-based index into stripped text
#                 colno = offset - 1 - spaces
#                 end_colno = end_offset - 1 - spaces
#                 if colno >= 0:
#                     # non-space whitespace (likes tabs) must be kept for alignment
#                     caretspace = ((c if c.isspace() else ' ') for c in ltext[:colno])
#                     yield '    {}{}'.format("".join(caretspace), ('^' * (end_colno - colno) + "\n"))
#         msg = self.msg or "<no detail available>"
#         yield "{}: {}{}\n".format(stype, msg, filename_suffix)

#     def format(self, *, chain=True, _ctx=None):
#         """Format the exception.

#         If chain is not *True*, *__cause__* and *__context__* will not be formatted.

#         The return value is a generator of strings, each ending in a newline and
#         some containing internal newlines. `print_exception` is a wrapper around
#         this method which just prints the lines to a file.

#         The message indicating which exception occurred is always the last
#         string in the output.
#         """

#         if _ctx is None:
#             _ctx = _ExceptionPrintContext()

#         output = []
#         exc = self
#         if chain:
#             while exc:
#                 if exc.__cause__ is not None:
#                     chained_msg = _cause_message
#                     chained_exc = exc.__cause__
#                 elif (exc.__context__  is not None and
#                       not exc.__suppress_context__):
#                     chained_msg = _context_message
#                     chained_exc = exc.__context__
#                 else:
#                     chained_msg = None
#                     chained_exc = None

#                 output.append((chained_msg, exc))
#                 exc = chained_exc
#         else:
#             output.append((None, exc))

#         for msg, exc in reversed(output):
#             if msg is not None:
#                 yield from _ctx.emit(msg)
#             if exc.exceptions is None:
#                 if exc.stack:
#                     yield from _ctx.emit('Traceback (most recent call last):\n')
#                     yield from _ctx.emit(exc.stack.format())
#                 yield from _ctx.emit(exc.format_exception_only())
#             elif _ctx.exception_group_depth > self.max_group_depth:
#                 # exception group, but depth exceeds limit
#                 yield from _ctx.emit(
#                     f"... (max_group_depth is {self.max_group_depth})\n")
#             else:
#                 # format exception group
#                 is_toplevel = (_ctx.exception_group_depth == 0)
#                 if is_toplevel:
#                     _ctx.exception_group_depth += 1

#                 if exc.stack:
#                     yield from _ctx.emit(
#                         'Exception Group Traceback (most recent call last):\n',
#                         margin_char = '+' if is_toplevel else None)
#                     yield from _ctx.emit(exc.stack.format())

#                 yield from _ctx.emit(exc.format_exception_only())
#                 num_excs = len(exc.exceptions)
#                 if num_excs <= self.max_group_width:
#                     n = num_excs
#                 else:
#                     n = self.max_group_width + 1
#                 _ctx.need_close = False
#                 for i in range(n):
#                     last_exc = (i == n-1)
#                     if last_exc:
#                         # The closing frame may be added by a recursive call
#                         _ctx.need_close = True

#                     if self.max_group_width is not None:
#                         truncated = (i >= self.max_group_width)
#                     else:
#                         truncated = False
#                     title = f'{i+1}' if not truncated else '...'
#                     yield (_ctx.indent() +
#                            ('+-' if i==0 else '  ') +
#                            f'+---------------- {title} ----------------\n')
#                     _ctx.exception_group_depth += 1
#                     if not truncated:
#                         yield from exc.exceptions[i].format(chain=chain, _ctx=_ctx)
#                     else:
#                         remaining = num_excs - self.max_group_width
#                         plural = 's' if remaining > 1 else ''
#                         yield from _ctx.emit(
#                             f"and {remaining} more exception{plural}\n")

#                     if last_exc and _ctx.need_close:
#                         yield (_ctx.indent() +
#                                "+------------------------------------\n")
#                         _ctx.need_close = False
#                     _ctx.exception_group_depth -= 1

#                 if is_toplevel:
#                     assert _ctx.exception_group_depth == 1
#                     _ctx.exception_group_depth = 0