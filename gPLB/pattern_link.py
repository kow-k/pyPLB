## imports

try:
    from .utils import *
except ImportError:
    from utils import *
try:
    from .pattern import *
except ImportError:
    from pattern import *

##
def make_PatternLinks_ranked(L, check: bool = False):
    "takes a lis to PatternLinks and returns a dictionary of {rank: [link1, link2, ...]}"
    ranked_links = {}
    for link in L:
        rank = link.get_rank()
        try:
            if not link in ranked_links[rank]:
                ranked_links[rank].append(link)
        except KeyError:
            ranked_links[rank] = [link]
    ##
    return ranked_links

##
class PatternLink:
    "definition of PatternLink class"
    def __init__ (self, pair, link_type = None):
        assert len (pair) == 2
        left, right = pair[0], pair[1]
        #assert len (left) == len (right) # offensive
        self.left            = left
        self.right           = right
        self.link_type       = link_type
        self.paired          = (left, right)
        self.form_paired     = (left.form, right.form)
        self.content_paired  = (left.content, right.content)
        self.gap_mark        = left.gap_mark

    ##
    def __len__(self):
        #assert len(self.left) == len(self.right)
        assert len(self.left) >= len(self.right)
        return max(len(self.left), len(self.right))

    ##
    def __repr__ (self):
        return f"{type(self).__name__} (\nL: {self.left};\nR: {self.right};\ntype: {self.link_type})"

    ##
    def __iter__ (self):
        for x in self.paired:
            yield x

    ##
    def get_rank (self):
        "takes a PatternLink and returns the rank of it"
        left, right = self.left, self.right
        gap_mark    = self.gap_mark
        #assert len(left) == len(right)
        assert len(left) >= len(right)
        form = left.form
        return len([ x for x in form if x != gap_mark ])

    ##
    def print (self, indicator = None, link_type = None, condition = None , paired: bool = False, check: bool = False):
        """
        prints the content of PatternLink object.
        condition can be a lambda expression used to filter.
        """
        ##
        if check:
            print(f"# self: {self!r}")
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
            arrow = "<--"
        elif link_type in [ "schematizes", "schematization", "part-of", "has-a" ]:
            arrow = "-->"
        else:
            arrow = "<-->"
        ##
        if paired:
            out = (f"{p_index}{p.form} {arrow} {q.form} // {p.content} {arrow} {q.content}")
        else:
            out = (f"{p_index}{p.form} {arrow} {q.form}")
        #
        print (out)

### end of file
