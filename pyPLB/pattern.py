## imports
#import array # turned out not to be suited
#import numpy as np # turned out not to be suited
#import awkward as ak # turned out not to be suited

## Functions
##
def list_encode_for_pattern (L: list) -> list:
    """
    take a list L of segments and returns a list R of (form, content) tuples
    R needs to be a list because it needs to be expanded later at building generalized PatternLattice
    """
    # Crucially, strip(..)
    return [ (str(x).strip(), [str(x).strip()]) for x in L if len(x) > 0 ]


##
def wrapped_merger_main (args):
    "utility function for Pool in merge_patterns"
    return merger_main (*args)

##
def check_for_instantiation (self, other, check: bool = False):
    "tests the instantiation of a pair of pattern with the equal size"
    r, l = self, other
    try:
        assert len(l) >= len(r)
    except AssertionError:
        return False
    ##
    gap_mark         = self.gap_mark
    tracer           = self.tracer
    r_form           = self.form
    r_untraced_form  = [ seg.replace(tracer, "") for seg in self.form ]
    l_form           = other.form
    l_untraced_form  = [ seg.replace(tracer, "") for seg in other.form ]
    ## filter invalid cases
    if r_form == l_form or r_form == l_untraced_form:
        if check:
            print(f"#is-a:F0; {l_form} ~~ {r_form}")
        return False
    ## Crucial for generating hierarchical organization!
    if abs(r.get_substance_size() - l.get_substance_size()) > 1:
        if check:
            print(f"#is-a:F1; {l_form} ~~ {r_form}")
        return False
    ##
    for i, r_seg in enumerate (r_form):
        l_seg = l_form[i]
        l_untraced_seg = l_untraced_form[i]
        if l_seg == r_seg or l_untraced_seg == r_seg:
            pass
        else: # l_seg != r_seg
            if l_seg != gap_mark:
                if check:
                    print(f"#is-a:F2; {l_form} ~~ {r_form}")
                return False
            elif l_seg != r_seg:
                if check:
                    print(f"#is-a:F3; {l_form} ~~ {r_form}")
                False
            else:
                pass
    ##
    if check:
        print(f"#is-a:T1; {l_form} <- {r_form}")

    #yield True # offensive??
    return True

## aliases
#test_for_is_a_relation = check_instantiation

##
def test_completion_of_gap_creation (P: list, gap_mark: str, check: bool = False) -> bool:
    "tests if P, a list of patterns, contains a fully gapped pattern"
    for p in P:
        if check:
            print(f"#checking for completion: {p}")
        if p.is_fully_gapped (gap_mark = gap_mark): # Crucially, len(x) > 0
            return True
    return False

##
def gen_L1_generalized_nodes (L, check: bool = False):
    """
    creates nodes at level 1 generalization to a given L
    """
    G = []
    for p in L:
        for position in [ 'left', 'right', 'both' ]:
            g = p.add_gaps_at_edge (position)
            if check:
                print(f"g: {g}")
            if g not in G:
                G.append(g)
    return G

def gen_L2_generalized_nodes (L, check: bool = False):
    """
    creates nodes at level 2 generalization to a given L
    """
    R = []
    for p in L:
        for i, g in enumerate(p.create_gap_inserted_versions ()):
            if g not in R:
                R.append(g)
    return R

### The idea of using NamedTuple turned out inadequate
#from collections import UserList
#class Pattern(UserList): # uses UserList
class Pattern:
    """
    Definitions for Pattern object
    """
    def __init__ (self, L: (list, tuple), gap_mark: str, tracer: str, boundary_mark: str = '#', check: bool = False):
        """
        creates a Pattern object from a given L, a list of elements, or from a paired
        """

        self.paired        = list_encode_for_pattern (L)
        self.gap_mark      = gap_mark
        self.tracer        = tracer
        self.boundary_mark = boundary_mark
        ## form
        #self.form          = [ x[0] for x in self.paired ]
        #self.form          = ak.Array([ x[0] for x in self.paired ]) # not work
        #self.form          = tuple([ x[0] for x in self.paired ]) # works
        self.form          = [ x[0] for x in self.paired ] # stopped using tuple 2025/01/05
        ## form_hash
        self.form_hash     = hash(tuple(self.form))
        ## content
        #self.content       = [ x[1] for x in self.paired ]
        #self.content       = ak.Array([ x[1] for x in self.paired ]) # not work
        #self.content       = tuple([ x[1] for x in self.paired ]) # works
        self.content       = [ x[1] for x in self.paired ] # stopped using tuple 2025/01/05
        ## size and others
        self.size          = len (self.form)
        self.rank          = self.get_rank()
        self.gap_size      = self.get_gap_size()
        self.content_size  = self.get_substance_size()
        #return self # offensive

    ## This is crucial
    def __eq__ (self, other):
        "defines response to '==' operator"
        # Multi-step returns get the judgement significantly faster
        #if self.form_hash != other.form_hash:
        #    return False
        #if self is None or other is None:
        #    return False
        if len (self.form) != len (other.form):
            return False
        if self.form != other.form:
            return False
        #elif self.content != other.content:
        #    return False
        else:
            return True

    ## define response to Pattern(None) is None
    def __bool__ (self):
        "defines response to Pattern(None) as None"
        return self.paired is not None

    ##
    def __len__ (self):
        "defines response to len()"
        try:
            return len (self.paired)
        except TypeError:
            return len (self.form)

    ##
    def __lt__ (self, other):
        #return len(self.form) < len(other.form)
        return self.form < other.form # tricky

    ## Basic list-like properties
    #def __get__item (self, index): # This was a mistake
    def __getitem__ (self, index):
        return self.paired[index]
    ##
    #def __set__item (self, index, value): # This was a mistake
    def __setitem__ (self, index, value):
        "defines the response to __setitem__, i.e., x[y] operator"
        self.paired[index] = value
        #return self # unnecessary?
    ##
    def __delitem__ (self, index):
        del self.paired[index]

    ## make iterable
    def __iter__ (self):
        return iter (self.paired)
        ## replaced the following
        #for x in self.paired: yield x

    ## define string representation
    def __repr__ (self):
        "defines response to print()"
        #return f"Pattern ({self.paired!r})" # is bad
        return f"{type(self).__name__} ({self.paired!r})"

    ##
    def copy (self):
        """
        implements Pattern.copy()
        """
        import copy
        C = []
        for segment in self.paired:
            c = copy.deepcopy(segment)
            C.append(c)
        return C

    ##
    def insert (self, index, value):
        self.paired.insert (index, value)
    ##
    def append (self, value):
        self.paired.append (value)
    ##
    def remove (self, value):
        self.paired.remove (value)
    ##
    def pop (self, index = -1):
        return self.paired.pop (index)

    ## Custom methods
    ##
    def separate_print (self, separator: str = " // "):
        return f"{type(self).__name__} ({self.form!r}{separator}{self.content!r})"

    ##
    def get_form (self):
        "takes a pattern and returns its form as list"
        return [ x[0] for x in self.form ]
        #return tuple([ x[0] for x in self.paired ]) # 2025/01/05
    ##
    def get_form_size (self):
        "takes a pattern and returns its form size"
        return len(self.get_form())
    ##
    def get_content (self):
        "takes a pattern and returns its content as a list"
        return [ x[1] for x in self.content ]
        #return tuple([ x[1] for x in self.content ])
    ##
    def get_content_size (self):
        "takes a pattern and returns its content size"
        return len(self.get_content())
    ##
    def get_substance (self):
        "takes a pattern and returns the list of non-gap elements in it"
        return [ x for x in self.form if x != self.gap_mark]
        #return tuple ([ x for x in self.form if x != self.gap_mark ])
    ##
    def get_substance_size (self):
        "takes a pattern and returns its rank, i.e., the number of non-gap elements"
        #return len([ x for x in self.form if x != self.gap_mark ])
        return len (self.get_substance())
    ## alias
    get_rank = get_substance_size

    ##
    def get_gaps (self):
        "takes a pattern and returns the list of gaps"
        return [ x for x in self.form if x == self.gap_mark ]

    ##
    def get_gap_size (self):
        "takes a pattern and returns the number of gap_marks in it"
        return len (self.get_gaps())
    ## alias
    count_gaps = get_gap_size

    ##
    def includes (self, other, check: bool = False):
        """
        takes two patterns, A and B, and tests if A includes B.
        """
        self_substance  = self.get_substance()
        other_substance = other.get_substance()
        for x in other_substance:
            if not x in self_substance:
                return False
        return True

    ##
    def group_patterns_by_size (self: object, reverse: bool = False) -> dict:
        "takes a list of Patterns and returns a dict whose keys are sizes of them"
        ##
        L = self.nodes
        sized_dict = {}
        for size in sorted (set ([ len(p.form) for p in L ]), reverse = reverse):
            sized_dict[size] = [ p for p in L if len (p.form) == size ]
        ##
        return sized_dict

    ##
    def has_compatible_content (R, L: list, check: bool = False) -> bool:
        "tests if a pair of Patterns has compatible contents"
        if check:
            print(f"#checking for content compatibility:\n{L}\n{R}")
        ## when input is a Pattern pair
        try:
            L_content, R_content = L.content, R.content
            if len(L_content) != len(R_content):
                return False
            for i, x in enumerate(L_content):
                if set(x) == set(R_content[i]):
                    pass
                else:
                    return False
            #
            return True
        ## when input is a list pair
        except AttributeError:
            if len(L) != len(R):
                return False
            for i, x in enumerate(L):
                if set(x) == set(R[i]):
                    pass
                else:
                    return False
            #
            return True

    ##
    def update_with_paired (self, paired):
        q = Pattern([])
        q.gap_mark      = self.gap_mark
        q.boundary_mark = self.boundary_mark
        q.paired        = paired
        q.form          = q.get_form ()
        q.content       = q.get_content ()
        return q

    ##
    def update_form (self):
        "updates self.form value of a Pattern given"
        try:
            #self.form = tuple( [ x[0] for x in self.paired ] )
            self.form = [ x[0] for x in self.paired ]
            return self
        except (AttributeError, TypeError):
            return None

    #
    def update_content (self):
        "updates self.content value of a Pattern given"
        try:
            #self.content = tuple( [ x[1] for x in self.paired ] )
            self.content = [ x[1] for x in self.paired ]
            return self
        except TypeError:
            return None

    ##
    def update_paired (self):
        "updates self.paired value of a Pattern given"
        #self.paired = tuple( [ (x, y) for x, y in zip (self.form, self.content) ] )
        self.paired = [ (x, y) for x, y in zip (self.form, self.content) ]
        #return self

    ##
    def create_gapped_versions (self: list, check: bool = False) -> list:
        """
        create a list of gapped patterns in which each non-gap is replaced by a gap
        """
        if check:
            print(f"#self: {self}")
        ##
        gap_mark  = self.gap_mark
        tracer    = self.tracer
        paired    = self.paired
        # generate a lattice
        R = [ ]
        for i in range (len(paired)):
            gapped = list(paired) # creates a copy of list form
            #gapped = deepcopy(paired) # This makes a copy of tuple.
            #form    = [ x[0] for x in paired ]
            #content = [ x[1] for x in paired ]
            #form    = tuple([ x[0] for x in paired ])
            #content = tuple([ x[1] for x in paired ])
            form    = [ x[0] for x in paired ] # Stopped using tuple on 2025/09/24
            content = [ x[1] for x in paired ] # Stopped using tuple on 2025/09/24

            f, c = form[i], content[i]
            if f != gap_mark:
                gapped[i] = (gap_mark, c)
            ## conversion to tuple
            #gapped = tuple((x, y) for x, y in zip(form, content))
            if check:
                print(f"#gapped: {gapped}")
            ##
            result        = Pattern([], gap_mark = gap_mark, tracer = tracer)
            result.paired = gapped
            result.update_form()
            result.update_content()
            ##
            R.append(result)
        #
        return R

    ##
    def create_random_gaps (self, n: int, check: bool = False):
        "create n gaps in the form of a given pattern"
        sites = range(len(self.form))
        gapped_sites = random.sample(sites, n)
        if check:
            print(f"# gapped_sites: {gapped_sites}")
        form = [ ]
        for i, f in enumerate (self.form):
            if i in gapped_sites:
                form.append (self.gap_mark)
            else:
                form.append (f)
        self.form = form
        if check:
            print(f"# self.form: {self.form}")
        self.update_paired()
        if check:
            print(f"# self.paired: {self.paired}")
        return self # Crucially

    ##
    def is_fully_gapped (self, gap_mark: str):
        "tests if a given pattern has the fully gapped form"
        #if all([ x == gap_mark for x in self.form if len(x) > 0 ]):
        #    return True
        ## The following replaced the above
        for seg in self.form:
            if len(seg) > 0 and seg != gap_mark:
                return False
        return True

    ##
    def add_gaps_at_edge (self, position: str, gap_content: str = "_", check: bool = False):
        "add a gap at edge of a pattern given"

        gap_mark     = self.gap_mark
        gapped_seg  = (gap_mark, [gap_content])
        #paired_new = self.paired[:] # causes trouble
        paired_new   = self.paired.copy() # Crucially
        if position in [ 'R', 'Right', 'right', 'r' ]:
            paired_new.append (gapped_seg)
        else:
            paired_new.insert(0, gapped_seg)
            if position in [ 'L', 'Left', 'left', 'l' ]:
                pass
            elif position in [ 'B', 'Both', 'both', 'b' ]:
                paired_new.append (gapped_seg)
            else:
                raise "Specified position is undefined"

        ## create a new pattern and update with the process above
        result = Pattern([], gap_mark = gap_mark)
        result.paired = paired_new
        result.update_form()
        result.update_content() # Don't forget to add () at the end!
        return result

    ##
    def create_gap_inserted_versions (self, gap_content: str = "_", check: bool = False):
        "add a gap at edge of a pattern given"

        n = len(self)
        if n < 2:
            return self
        else:
            #input_p = self[:] # causes trouble
            input_p = self.copy()
        ##
        gap_mark = self.gap_mark
        gapped_seg = (gap_mark, [gap_content])
        G = [] # holder of gapped versions
        import itertools
        positions = list(range(1, n)) # all positions to insert gaps
        # generate all non-empty subsets of positions
        for r in range(1, len(positions) + 1):
            for pos_combo in itertools.combinations(positions, r):
                if check:
                    print(f"pos_combo: {pos_combo}")
                #copied_p = input_p[:] # causes trouble
                copied_p = input_p.copy()
                if check:
                    print(f"copied_p: {copied_p}")
                # Insert gaps from right to left to maintain indices
                for pos in reversed (pos_combo):
                    if check:
                        print(f"pos: {pos}")
                    gapped_p = copied_p[:pos] + [gapped_seg] + copied_p[pos:]
                    ## make it Pattern, crucially
                    new_paired = [ (x[0], x[1]) for x in gapped_p ]
                    gapped_p = Pattern([], gap_mark = gap_mark)
                    gapped_p.paired = new_paired
                    gapped_p.update_form()
                    gapped_p.update_content()
                    if check:
                        print(f"gapped_p: {gapped_p}")
                if gapped_p not in G:
                    G.append(gapped_p)
        ## return result
        return G

    ##
    def transpose (p, trace_mark: str = "~", check: bool = False):
        """
        Implements transposition
        """
        r = p.copy()
        R = []
        for i in range(len(p)):
            x = p[i]
            for j in range(1,len(p)):
                if i != j:
                    r.insert(j, x)
                    R.append(r)
        ##
        return R

    ##
    def build_lattice_nodes (p, generalized: bool, fully_generalized: bool, check: bool = False):
        "takes a pattern and returns a list of lattice nodes"
        if check:
            print (f"#p: {p}")
        #
        gap_mark = p.gap_mark
        size     = len(p.paired)
        # main
        form_register = [p.form]
        completed = False
        R = [p] # holder of result
        while not completed:
            if check:
                print(f"#completed: {completed}")
            for r in R:
                if check:
                    print(f"#r: {r}")
                ## creates gapped versions of a given pattern
                G = r.create_gapped_versions (check = False)
                for i, g in enumerate (G):
                    if check:
                        print(f"#g{i}: {g}")
                    if g.form not in form_register:
                        R.append (g)
                        form_register.append(g.form)
            ## check termination
            if test_completion_of_gap_creation (R, gap_mark = gap_mark, check = False):
                completed = True

        ## The following block was replaced by input manipulation before building patterns
        ## build generalized lattice
        #if generalized:
        #    # level 2
        #    G2 = []
        #    if fully_generalized:
        #        for i, g in enumerate(gen_L2_generalized_nodes (R, check = check)):
        #            print(f"#added g{i} in G2: {g}")
        #        R.extend(G) # add G2 elements to R
        #    # level 1
        #    G1 = R.copy()
        #    for i, g in enumerate(gen_L1_generalized_nodes (R, check = check)):
        #        if g not in G1:
        #            print(f"#added g{i} in G1: {g}")
        #        G1.append(g)
        #    R = G1 # update R with G1 values
        #    if check:
        #        print(f"#R: {R}")

        ## return result
        return sorted (R)

    ##
    def instantiates_or_not (self, other, check: bool = False) -> bool:
        """
        tests if pattern R instantiates another L, i.e., instance(R, L) == part_of(L, R)
        """
        gap_mark        = self.gap_mark
        boundary_mark   = self.boundary_mark
        R, L = self, other
        #R_form, L_form = R.form, L.form
        R_form, L_form = self.form, other.form
        #R_size, L_size = R.size, L.size ## fails
        R_size, L_size = len(R.form), len(L.form)
        #R_rank, L_rank = R.rank, L.rank # fails
        R_rank, L_rank  = R.get_rank(), L.get_rank()
        R_content       = self.content
        L_content       = other.content
        R_content_size  = len(R_content)
        L_content_size  = len(L_content)
        R_substance     = R.get_substance()
        L_substance     = L.get_substance()
        if check:
            print(f"===================")
            print(f"#L: {L}; R: {R}")
            print(f"#L_size: {L_size}; R_size: {R_size}")
            print(f"#L_rank: {L_rank}; L_rank: {R_rank}")
            print(f"#L_form: {L_form}; R_form: {R_form}")
        ## L's rank is one-segment smaller than R's
        if L_size == R_size and L_rank + 1 == R_rank:
            return check_instantiation (R, L, check = check)
        ## when L is one-segment longer than R
        #elif L_size - 1 == R_size and L_rank == R_rank: # goes awry, but why?
        #elif (L_size + 1 == R_size) and len(L_substance) == len(R_substance):
        elif (L_size - 1 == R_size) and len(L_substance) == len(R_substance):
            if L_substance == R_substance:
                ## risks overgeneration ...
                return True
                if check:
                    print(f"L_substance: {L_substance}")
                    print(f"R_substance: {R_substance}")
            else:
                return False
        ## other cases
        else:
            return False

    ##
    def merge_patterns (self, other, reduction: bool = True, check: bool = False):
        "take a pair of Patterns, merges one Pattern with another"
        if check:
            print(f"#=====================")
            print(f"#self:  {self}")
            print(f"#other: {other}")
        ##
        gap_mark       = self.gap_mark
        boundary_mark  = self.boundary_mark
        ##
        if len(self) != len(other):
            return None
        ## prevents void operation
        if self.form == other.form and self.content == other.content:
            return self
        ## main
        form_pairs     = list(zip (self.form, other.form))
        content_pairs  = list(zip (self.content, other.content))
        if check:
            print(f"#form_pairs :{form_pairs}")
            print(f"#content_pairs: {content_pairs}")#
        ##
        new_paired = merge_patterns_with_equaly_size (form_pairs, content_pairs, gap_mark, boundary_mark, check = check)
        if check:
            print(f"#new_paired: {new_paired}")
        ##
        new = Pattern([], gap_mark = gap_mark)
        new.paired = new_paired
        new.update_form()
        new.update_content()
        #yield new # fails
        return new

### end of file
