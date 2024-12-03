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
##
class PatternLink:
    "definition of PatternLink class"
    def __init__ (self, pair, link_type: str = ""):
        assert len (pair) == 2
        left, right = pair[0], pair[1]
        #assert len (left) == len (right) # offensive
        self.left            = left
        self.right           = right
        self.gap_mark        = left.gap_mark
        self.link_type       = link_type
        self.paired          = (left, right)
        self.form_paired     = (left.form, right.form)
        self.content_paired  = (left.content, right.content)

    ## Unimplementation of this method seems the last cause for slow processing
    def __eq__ (self, other):
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
    def get_link_rank (self: object) -> int:
        "takes a PatternLink and returns the rank of it"
        left, right  = self.left, self.right
        gap_mark     = self.gap_mark
        #assert len(left) == len(right)
        #assert abs(len(left) - len(right)) <= 1
        #form = left.form
        #return len([ x for x in form if x != gap_mark ])
        l_size = len ([ x for x in left.form if x != gap_mark ])
        r_size = len ([ x for x in right.form if x != gap_mark ])
        #return min (l_size, r_size) # produces null merger
        return max (l_size, r_size)

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
