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
    return [ (str(x).strip(), [str(x).strip()] ) for x in L if len(x) > 0 ]

##
def is_tracing_pair (f: str, g: str, tracer: str, strict: bool = True) -> bool:
    """
    tests if a given pair of strings is a tracing pair in that one is the tracer of the other
    """
    if strict:
        if f[0] == tracer and f[1:] == g:
            return True
        else:
            return False
    else:
        if (f[0] == tracer and f[1:] == g) or (g[0] == tracer and f == g[1:]):
            return True
        else:
            return False

##
def test_gapping_completion (P: list, gap_mark: str, check: bool = False) -> bool:
    "tests if P, a list of patterns, contains a fully gapped pattern"
    for p in P:
        if check:
            print(f"#checking for completion: {p}")
        if p.is_fully_gapped (gap_mark = gap_mark): # Crucially, len(x) > 0
            return True
    return False

##
def isa_under_size_equality (r_form, l_form, gap_mark: str, tracer: str = '~', check: bool = False) -> bool:
    """
    checks if r_form instantiates l_form under size equality, returning True or False
    """

    ## handling type mismatch
    if r_form is Pattern:
        r_form = r_form.form
    if l_form is Pattern:
        l_form = l_form.form

    ##
    for i, r_seg in enumerate (r_form):
        l_seg = l_form[i]
        ## compare r_seg and l_seg and return False if isa fails to hold
        if l_seg == r_seg:
            pass
        elif is_tracing_pair (l_seg, r_seg, tracer = tracer):
            pass
        else:
            if l_seg == gap_mark:
                pass
            else:
                if r_seg != gap_mark and l_seg != r_seg:
                    print(f"#is-a: {l_form} ~~ {r_form}")
                    return False
                else:
                    pass
    #if check:
    print(f"#is-a: {l_form} <= {r_form}")
    return True

##
def gen_L1_generalized_nodes (L, check: bool = False):
    """
    creates nodes at level 1 generalization to a given L
    """
    G = []
    for p in L:
        for position in [ 'left', 'right', 'both' ]:
            g = p.add_gaps_around (position)
            if check:
                print(f"g: {g}")
            if g not in G:
                G.append(g)
    return G

def gen_L2_generalized_nodes (L, check: bool = False):
    """
    creates nodes at level 2 generalization to a given L
    """
    G = []
    for p in L:
        for i, g in enumerate(p.create_gap_inserted_versions ()):
            if check:
                print(f"inserting g{i}: {g}")
            if g not in G:
                G.append(g)
    return G

##
def merge_patterns_with_equal_size (form_pairs: list, content_pairs: list, gap_mark: str, boundary_mark: str, check: bool = False):
    ## The following operation needs to be re-implemented for speed up
    #import numpy as np
    ##
    Fa = [ f[0] for f in form_pairs ]
    Fb = [ f[1] for f in form_pairs ]
    if abs(get_rank(Fa) == get_rank(Fb)) > 1:
        return None
    ##
    new_form    = [ ]
    void_result = [ ]
    new_content = [ [ ] for _ in range(len(form_pairs)) ] # Crucially
    for i, pair in enumerate (form_pairs):
        fa, fb = pair[0], pair[1]
        ca, cb = content_pairs[i]
        C = [ x[0] for x in content_pairs[i] ] # Crucially
        ## handles form
        if fa is None or fb is None:
            return None
            #return void_result # Crucially
        elif fa == fb:
            new_form.append (fa)
            new_content[i].extend (C)
        else:
            if fa == gap_mark:
                new_form.append (fb)
                if ca == cb:
                    new_content[i].extend(C)
                else:
                    ## handling boundary marking
                    if ca == boundary_mark:
                        new_content[i].append (cb)
                    else:
                        return None
                        #return void_result # Crucially
            elif fb == gap_mark:
                new_form.append (fa)
                if ca == cb:
                    new_content[i].extend (C)
                else:
                    ## handling boundary marking
                    if cb == boundary_mark:
                        new_content[i].append (ca)
                    else:
                        return None
                        #return void_result # Crucially
            else:
                return None
                #return void_result # Crucially
    ## Cruially, list(...)
    new_paired  = [ (F, C) for F, C in list(zip(new_form, new_content)) ]
    #yield new_paired # fails
    return new_paired

##
def wrapped_merger_main (args):
    "utility function for Pool in merge_patterns"
    return merger_main (*args)

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
        self.form          = [ x[0] for x in self.paired ] # revived on 2025/01/05
        #self.form          = ak.Array([ x[0] for x in self.paired ]) # not work
        #self.form          = tuple([ x[0] for x in self.paired ]) # works

        ## form_hash
        self.form_hash     = hash(tuple(self.form))

        ## content
        self.content       = [ x[1] for x in self.paired ] # revived on 2025/01/05
        #self.content       = ak.Array([ x[1] for x in self.paired ]) # not work
        #self.content       = tuple([ x[1] for x in self.paired ]) # works

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
        #return len(self.form) < len(other.form) # harmful
        #return self.form < other.form # tricky
        try:
            return self.form < other.form # tricky
        except TypeError:
            return tuple(self.form) < tuple(other.form)

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
        #return tuple([ x[0] for x in self.paired ]) # 2025/01/05
        return [ x[0] for x in self.form ]

    ##
    def get_form_size (self):
        "takes a pattern and returns its form size"
        return len(self.get_form())
    ## alias
    get_size = get_form_size

    ##
    def get_content (self):
        "takes a pattern and returns its content as a list"
        #return tuple([ x[1] for x in self.content ]) # stopped using this
        return [ x[1] for x in self.content ]

    ##
    def get_content_size (self):
        "takes a pattern and returns its content size"
        return len(self.get_content())
    ##
    def get_substance (self):
        "takes a pattern and returns the list of non-gap elements in it"
        #return [ x for x in self.form if x != self.gap_mark and x[0] != self.tracer ]
        return [ x for x in self.form if x != self.gap_mark ]

    ##
    def get_g2_substance (self):
        "takes a pattern and returns the list of non-gap elements in it"
        return [ x for x in self.form if x != self.gap_mark and x[0] != self.tracer ]

    ##
    def get_substance_size (self):
        "takes a pattern and returns its rank, i.e., the number of non-gap elements"
        return len (self.get_substance())
    ## alias
    get_rank = get_substance_size

    ##
    def get_g2_substance_size (self):
        "takes a pattern and returns its rank, i.e., the number of non-gap elements"
        return len (self.get_g2_substance())
    ## alias
    get_g2_rank = get_g2_substance_size

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
            form    = [ x[0] for x in paired ] # stopped using tuple on 2025/09/24
            content = [ x[1] for x in paired ] # stopped using tuple on 2025/09/24

            f, c = form[i], content[i]
            if f != gap_mark:
                gapped[i] = (gap_mark, c)
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
    def add_gaps_around (self, position: str, gap_content: str = "_", check: bool = False):
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
    def merges_with (self, other, reduction: bool = True, check: bool = False):
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
        new_paired = merge_patterns_with_equal_size (form_pairs, content_pairs, gap_mark, boundary_mark, check = check)
        if check:
            print(f"#new_paired: {new_paired}")

        ## create Pattern for return
        new = Pattern([], gap_mark = gap_mark)
        new.paired = new_paired
        new.update_form()
        new.update_content()
        return new # yield new fails

    ##
    def build_lattice_nodes (p, generalized: bool, more_generalized: bool, check: bool = False):
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
            if test_gapping_completion (R, gap_mark = gap_mark, check = False):
                completed = True

        ## build generalized lattice
        #if generalized:
        #    # level 2
        #    G2 = []
        #    if more_generalized:
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
    def instantiates_or_not_rev (self, other, check: bool = False) -> bool:
        """
        tests if pattern R instantiates another L, i.e., instance(R, L) == part_of(L, R)
        """
        gap_mark        = self.gap_mark
        boundary_mark   = self.boundary_mark

        R, L = self, other
        R_form, L_form  = self.form, other.form
        R_size, L_size  = len(R.form), len(L.form)
        R_rank, L_rank  = R.get_rank(), L.get_rank()
        R_gap_size, L_gap_size  = R.form.get_gap_size(), L.form.get_gap_size()
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

        ##
        size_diff = L_size - R_size
        gap_diff  = L_gap_size - R_gap_size
        rank_diff = R_rank - L_rank

        ## L and R have a size difference more than 1
        if abs(size_diff) > 1:
            return False

        ## L and R have the same size and L's rank is one-segment smaller than R's
        elif size_diff == 0:
            return isa_under_size_equality (R_form, L_form, gap_mark = gap_mark, check = check)

        ## when L is one-segment longer than R
        ## This case needs revision to handle generalizeation level 2
        elif size_diff == 1 and rank_diff == 1:
            if L_gap_size == R_gap_size + 1:
                if L_substance == R_substance:
                    ## risks overgeneration ...
                    if check:
                        print(f"L_substance: {L_substance}")
                        print(f"R_substance: {R_substance}")
                    return True
                else:
                    if check:
                        print(f"L_substance: {L_substance}")
                        print(f"R_substance: {R_substance}")
                    return False
            else:
                if check:
                    print(f"L_substance: {L_substance}")
                    print(f"R_substance: {R_substance}")
                return False

        ## other cases
        else:
            return False

    ##
    def instantiates_or_not_prev (self, other, check: bool = False) -> bool:
        """
        tests if pattern R instantiates another L, i.e., instance(R, L) == part_of(L, R)
        """
        gap_mark        = self.gap_mark
        boundary_mark   = self.boundary_mark
        R, L = self, other
        R_form, L_form = self.form, other.form
        R_size, L_size = len(R.form), len(L.form)
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

        ##
        size_diff = L_size - R_size
        ## L and R have a size difference more than 1
        if abs(size_diff) > 1:
            return False
        ## L and R have the same size and L's rank is one-segment smaller than R's
        elif L_size == R_size and L_rank == R_rank - 1:
        #elif L_size == R_size: # fails
            return isa_under_size_equality (R_form, L_form, gap_mark = gap_mark, check = check)
        ## when L is one-segment longer than R
        ## This case needs revision to handle generalizeation level 2
        #elif (L_size == R_size + 1) and len(L_substance) == len(R_substance):
        elif (L_size == R_size + 1):
            if L_gap_size == R_gap_size + 1:
                if L_substance == R_substance:
                    ## risks overgeneration ...
                    if check:
                        print(f"L_substance: {L_substance}")
                        print(f"R_substance: {R_substance}")
                    return True
                else:
                    if check:
                        print(f"L_substance: {L_substance}")
                        print(f"R_substance: {R_substance}")
                    return False
            else:
                if check:
                    print(f"L_substance: {L_substance}")
                    print(f"R_substance: {R_substance}")
                return False
        ## other cases
        else:
            return False

    ## alias
    instantiates_or_not = instantiates_or_not_prev

### end of file
