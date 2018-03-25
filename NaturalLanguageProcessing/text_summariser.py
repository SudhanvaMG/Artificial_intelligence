from  nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
from string import punctuation
from heapq import nlargest

class frequency_summarizer:
    def __init__(self,min_cut= 0.1,max_cut = 0.9):
        self.min_cut = min_cut
        self.max_cut = max_cut
        self.stop_words = set(stopwords.words("english")+list(punctuation))
    
    def compute_frequencies(self, word_sent):
        freq = defaultdict(int)
        for sent in word_sent:
            for word in sent:
                if word not in self.stop_words:
                    freq[word] += 1
    
        max_freq = float(max(freq.values()))
    
        for word in freq.keys():
            freq[word] = freq[word] / max_freq
            if freq[word] > self.max_cut or freq[word] < self.min_cut:
                del freq[word]
    
        return freq
    
    def summarize(self, text, n):
        sents = sent_tokenize(text)
        word_sent = [word_tokenize(s.lower()) for s in sents]
        self.freq = self.compute_frequencies(word_sent)
        rankings = defaultdict(int)
        for i, sent in enumerate(word_sent):
            for word in sent:
                if word in self.freq:
                    rankings[i] += self.freq[word]
    
        sent_idx  = nlargest(n,rankings, key = rankings.get)
    
        return [sents[j] for j in sent_idx]

#text = """ place your text here ---
#It took the talent, grit and artistry of eight American figure skaters over three days of competition at
# the Gangneung Ice Arena to deliver the United States a bronze medal in the team event at
# the PyeongChang Olympics.
#But the most magical and pivotal contribution came from Mirai Nagasu, and it was over in an instant. 
#In the span of one glorious eyeblink, Nagasu, who had been snubbed by U.S. skating officials for a 
#spot on the 2014 Olympic team, poured all she had worked toward these past four years into the opening 
#jump of her free skate on the final day of the team competition.
#
#And when she landed solidly on one foot, after making 3½ rotations in the air, 
#Nagasu made history, becoming the first American woman to land the high-risk triple axel in Olympic
# competition.
#
#
#
#That she did, contributing nine valuable points to the United States’ bronze medal effort.
#
#
#Canada, which boasts the world’s top ice dance pair, won gold. The Olympic Athletes from Russia took 
#silver, giving the motherland they are forbidden from acknowledging at these Olympics its second medal.
# Under International Olympic Committee sanctions following evidence of state-sponsored doping at the
# 2014 Sochi Games, Russia was banned from these Olympics but, in a compromise, was allowed to send 
# 168 athletes absolved of any part in the scandal to compete under a stateless “OAR” banner.
#
#
#The unprecedented compromise meant that during the flower ceremony that followed Monday’s competition, 
#OAR silver medalists ascended the silver stand on the podium in drab gray and red warm-ups and were
# forbidden from displaying a Russian flag while Canadian and American skaters wore their colors with
# pride.
#"""

fs = frequency_summarizer()
fs.summarize(text,3)




    
