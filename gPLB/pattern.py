## imports
#import array # turned out not to be suited
#import numpy as np # turned out not to be suited
#import awkward as ak # turned out not to be suited

## Functions
def list_encode_for_pattern (L: list) -> list:
    """
    take a list L of segments and returns a list R of (form, content) tuples
    R needs to be a list because it needs to be expanded later at building generalized PatternLattice
    """
    # Crucially, strip(..)
    return [ (str(x).strip(), [str(x).strip()]) for x in L if len(x) > 0 ]

##
def test_completion (P: list, gap_mark: str, check: bool = False) -> bool:
    "tests if P, a list of patterns, contains a fully gapped pattern"
    for p in P:
        if check:
            print(f"#checking for completion: {p}")
        if p.is_fully_gapped (gap_mark = gap_mark): # Crucially, len(x) > 0
            return True
    return False


##
def merge_patterns_with_equaly_size (form_pairs: list, content_pairs: list, gap_mark: str, boundary_mark: str, check: bool = False):
    ## The following operation needs to be re-implemented for speed up
    #import numpy as np
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

##
def check_instantiation (self, other, check: bool = False):
    "tests the instantiation of a pair of pattern with the equal size"
    R, L = self, other
    try:
        assert len(L) >= len(R)
    except AssertionError:
        return False
    ##
    gap_mark   = self.gap_mark
    R_form     = self.form
    L_form     = other.form
    ##
    for i, r_seg in enumerate(R_form):
        l_seg = L_form[i]
        if l_seg == r_seg:
            pass
        else: # l_seg != r_seg
            if l_seg != gap_mark:
                if check:
                    print(f"#no instantiation with {L_form}; {R_form}")
                return False
            elif l_seg != r_seg:
                if check:
                    print(f"#no instantiation with {L_form}; {R_form}")
                False
            else:
                pass
    if check:
        print(f"#{R_form} instantiates {L_form}")
    ##
    #yield True # offensive??
    return True

##
class Pattern:
    # The idea of using NamedTuple turned out inadequate
    #def __init__ (L): # <= This is wrong.
    def __init__ (self, L: (list, tuple), gap_mark: str, boundary_mark: str = "#"):
        "creates a Pattern object from a given L, a list of elements, or from a paired"
        ##
        self.gap_mark      = gap_mark
        self.boundary_mark = boundary_mark
        ## paired
        self.paired        = list_encode_for_pattern (L)
        #self.paired        = tuple_encode_for_pattern (L) # not work
        ## form
        #self.form          = [ x[0] for x in self.paired ]
        #self.form          = ak.Array([ x[0] for x in self.paired ]) # not work
        self.form          = tuple([ x[0] for x in self.paired ]) # works
        ## form_hash
        self.form_hash     = hash(tuple(self.form))
        ## content
        #self.content       = [ x[1] for x in self.paired ]
        #self.content       = ak.Array([ x[1] for x in self.paired ]) # not work
        self.content       = tuple([ x[1] for x in self.paired ]) # works
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
        #return self.form < other.form
        return len(self.form) < len(other.form)

    ##
    def __iter__ (self):
        for x in self.paired:
            yield x

    ##
    def __repr__ (self):
        "defines response to print()"
        #return f"Pattern ({self.paired!r})" # is bad
        return f"{type(self).__name__} ({self.paired!r})"

    ##
    def __get__item (self, position):
        return self.paired[position]

    ##
    def __set__item (self, position, value):
        "defines the response to __setitem__, i.e., x[y] operator"
        self.paired[position] = value
        return self

    ##
    def get_form (self):
        "takes a pattern and returns its form as list"
        #return [ x[0] for x in self.form ]
        return tuple([ x[0] for x in self.form ])

    ##
    def get_form_size (self):
        "takes a pattern and returns its form size"
        return len(self.get_form())

    ##
    def get_content (self):
        "takes a pattern and returns its content as a list"
        #return [ x[1] for x in self.content ]
        return tuple([ x[1] for x in self.content ])

    ##
    def get_content_size (self):
        "takes a pattern and returns its content size"
        return len(self.get_content())

    ##
    def get_substance (self):
        "takes a pattern and returns the list of non-gap elements in it"
        #return [ x for x in self.form if x != self.gap_mark]
        return tuple ([ x for x in self.form if x != self.gap_mark])

    ##
    def get_substance_size (self):
        "takes a pattern and returns its rank, i.e., the number of non-gap elements"
        #return len([ x for x in self.form if x != self.gap_mark ])
        return len (self.get_substance())

    ## alias
    get_rank = get_substance_size

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
            #self.form = [ x[0] for x in self.paired ]
            self.form = tuple( [ x[0] for x in self.paired ] )
            return self
        except (AttributeError, TypeError):
            return None

    #
    def update_content (self):
        "updates self.content value of a Pattern given"
        try:
            #self.content = [ x[1] for x in self.paired ]
            self.content = tuple( [ x[1] for x in self.paired ] )
            return self
        except TypeError:
            return None

    ##
    def update_paired (self):
        "updates self.paired value of a Pattern given"
        #self.paired = [ (x, y) for x, y in zip (self.form, self.content) ]
        self.paired = tuple( [ (x, y) for x, y in zip (self.form, self.content) ] )
        return self

    ##
    def get_gaps(self):
        "takes a pattern and returns the list of gaps"
        return [ x for x in self.form if x == self.gap_mark ]

    ##
    def get_gap_size(self):
        "takes a pattern and returns the number of gap_marks in it"
        return len(self.get_gaps())

    ##
    def create_gapped_versions (self: list, check: bool = False) -> list:
        "create a list of gapped patterns in which each non-gap is replaced by a gap"
        if check:
            print(f"#self: {self}")
        ##
        gap_mark = self.gap_mark
        paired   = self.paired
        # generate a lattice
        R = [ ]
        for i in range (len(paired)):
            gapped = list(paired) # creates a copy of list form
            #gapped = deepcopy(paired) # This makes a copy of tuple.
            #form    = [ x[0] for x in paired ]
            #content = [ x[1] for x in paired ]
            form    = tuple([ x[0] for x in paired ])
            content = tuple([ x[1] for x in paired ])
            f, c = form[i], content[i]
            if f != gap_mark:
                gapped[i] = (gap_mark, c)
            ## conversion to tuple
            #gapped = tuple((x, y) for x, y in zip(form, content))
            if check:
                print(f"#gapped: {gapped}")
            ##
            result        = Pattern([], gap_mark = gap_mark)
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
    def add_gap_at_edge (self, position: str, edge_value: str = "#", check: bool = False):
        "add a gap at edge of a pattern given"
        gap_mark     = self.gap_mark
        paired_new   = self.paired.copy() # Crucially
        #paired_new   = self.paired # when tuple-encoding is applied
        gapped_edge  = (gap_mark, [edge_value])
        if position in [ 'Right', 'R', 'right' ]:
            paired_new.append (gapped_edge)
            #paired_new = tuple(paired_new, gapped_edge)
        else:
            paired_new.insert(0, gapped_edge)
            #paired_new = tuple(gapped_edge, paired_new)
            if position in [ 'Left', 'L', 'left' ]:
                pass
            elif position in [ 'Both', 'B', 'both' ]:
                paired_new.append (gapped_edge)
                #paired_new = tuple(paired_new, gapped_edge)
            else:
                raise "Specified position is undefined"
        ## return result
        result = Pattern([], gap_mark = gap_mark)
        result.paired = paired_new
        result.update_form()
        result.update_content() # Don't forget to add () at the end!
        return result

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
    def build_lattice_nodes (p, generalized: bool, check: bool = False):
        "takes a pattern and returns a list of lattice nodes"
        if check:
            print (f"#p: {p}")
        #
        gap_mark = p.gap_mark
        size     = len(p.paired)
        R        = [p]
        ## main
        form_register = [p.form]
        completed = False
        while not completed:
            if check:
                print(f"#completed: {completed}")
            for r in R:
                if check:
                    print(f"#r: {r}")
                G = r.create_gapped_versions (check = False)
                if check:
                    print(f"#G: {G}")
                for i, g in enumerate (G):
                    if check:
                        print(f"#g{i}: {g}")
                    if g.form not in form_register:
                        R.append (g)
                        form_register.append(g.form)
            ## check termination
            if test_completion (R, gap_mark = gap_mark, check = False):
                completed = True

        ## build generalized lattice
        if generalized:
            Q = R.copy()
            positions = [ 'right', 'left', 'both' ]
            for position in positions:
                for r in R:
                    q = r.add_gap_at_edge (position)
                    if q not in Q:
                        Q.append(q)
            if check:
                print(f"#Q: {Q}")
            R = Q
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
