## imports
try:
    from .utils import *
except ImportError:
    from utils import *
try:
    from .pattern import *
except ImportError:
    from pattern import *

## Functions

### Classes
class PatternLink:
    "definition of PatternLink class"
    def __init__ (self, pair, link_type: str = ""):

        """
        generator of PatternLink object
        """

        assert len (pair) == 2
        left_p, right_p = pair[0], pair[1]
        self.left            = left_p
        self.right           = right_p
        self.gap_mark        = left_p.gap_mark
        self.link_type       = link_type
        self.paired          = ( left_p, right_p )
        #self.form_paired     = ( tuple(left_p.form), tuple(right_p.form) )
        self.form_paired     = ( tuple(left_p.form), tuple(right_p.form) ) # 2025/10/13
        #self.content_paired  = ( left_p.content, right_p.content )
        self.content_paired  = ( tuple(left_p.content), tuple(right_p.content) ) # 2025/10/13

    ##
    def __hash__(self):
        """Make PatternLink hashable for efficient set/dict operations"""
        if not hasattr(self, '_hash_cache'):
            # Cache hash based on immutable tuple pairs
            self._hash_cache = hash((self.form_paired, self.content_paired))
        return self._hash_cache

    ##
    def __eq__ (self, other):
        if not isinstance(other, PatternLink):
            return False
        # Fast path: compare cached hashes first
        if hasattr(self, '_hash_cache') and hasattr(other, '_hash_cache'):
            if self._hash_cache != other._hash_cache:
                return False
        # Full comparison
        if len(self) != len(other):
            return False
        if self.left != other.left:
            return False
        else:
            if self.right != other.right:
                return False
            else:
                return True

    ## Unimplementation of this method seems the last cause for slow processing
    def __eq_old__ (self, other):
        if len(self) != len(other):
            return False
        if self.left != other.left:
            return False
        else:
            if self.right != other.right:
                return False
            else:
                return True

    ##
    def __len__ (self, use_min: bool = False):
        #assert len(self.left) == len(self.right)
        #assert len(self.left) >= len(self.right) # offensive
        #assert abs(len(self.left) - len(self.right)) < 2 # slows down
        if use_min:
            return min (len(self.left), len(self.right))
        else:
            return max (len(self.left), len(self.right))

    ##
    def __lt__(self, other):
        #return self.left < other.left and self.right < other.right
        return self.right < other.right or self.left < other.left

    ##
    def __repr__ (self):
        return f"{type(self).__name__} (l: {self.left}; r: {self.right};\ntype: {self.link_type})"

    ##
    def __iter__ (self):
        for x in self.paired:
            yield x

    ##
    def get_link_rank (self: object, use_max: bool = True) -> int:
        "takes a PatternLink and returns the rank of it"
        left, right  = self.left, self.right
        gap_mark     = self.gap_mark
        l_size = len ([ x for x in left.form if x != gap_mark ])
        r_size = len ([ x for x in right.form if x != gap_mark ])
        if use_max:
            return max (l_size, r_size)
        else:
            return min (l_size, r_size) # produces null merger

    ##
    def pprint (self, indicator = None, link_type = None, condition = None , paired: bool = False, pair_mark: str = "//", check: bool = False) -> None:
        """
        prints the content of PatternLink object.
        condition can be a lambda expression used to filter.
        """
        ##
        if check:
            print(f"#self: {self!r}")
        ##
        p, q = self.left, self.right
        ##
        if indicator:
            p_index = f"link {indicator:3d}: "
        else:
            p_index = ""
        ##
        if link_type is None:
            link_type = self.link_type
        if link_type in [ "instantiates", "instantiation", "is-a", "instance-of" ]:
            arrow = "->"
        elif link_type in [ "schematizes", "schematization", "part-of", "has-a" ]:
            arrow = "<-"
        else:
            arrow = "--"
        ##
        if paired:
            out = (f"{p_index}{p.form} {arrow} {q.form} {pair_mark} {p.content} {arrow} {q.content}")
        else:
            out = (f"{p_index}{p.form} {arrow} {q.form}")
        #
        print (out)

### end of file
