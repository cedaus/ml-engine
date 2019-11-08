import nltk

pos_family = {
    'noun' : ['NN','NNS','NNP','NNPS'],
    'pron' : ['PRP','PRP$','WP','WP$'],
    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
    'adj' :  ['JJ','JJR','JJS'],
    'adv' : ['RB','RBR','RBS','WRB']
}

NP_chunks = ['DT-JJ-NN']
VP_chunks = []
PP_chunks = []
ADJP_chunks = []
ADVP_chunks = []

chunks = {'DT-JJ-NN': '<DT>?<JJ>*<NN>'}

class GrammarConstructor():
    def __init__(self, text):
        self.text = text
        
    def get_pos(self, tokens):
        return nltk.pos_tag(tokens)
    
    def get_pos_tag_help(self, tag):
        print(nltk.help.upenn_tagset(tag))
        
    def get_count_for_pos_tags(self, pos):
        from collections import Counter
        return Counter([j for i,j in pos])