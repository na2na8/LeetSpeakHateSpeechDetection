import random
import sys

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import CountVectorizer

from gensim.models import KeyedVectors as W2Vec

from perturbations_store import PerturbationsStorage

class VIPER_DCES :
    def __init__(self, args) :
        # dces
        self.disallowed = ['TAG', 'MALAYALAM', 'BAMUM', 'HIRAGANA', 'RUNIC', 'TAI', 'SUNDANESE', 'BATAK', 'LEPCHA', 'CHAM',
                    'TELUGU', 'DEVANGARAI', 'BUGINESE', 'MYANMAR', 'LINEAR', 'SYLOTI', 'PHAGS-PA', 'CHEROKEE',
                    'CANADIAN', 'YI', 'LYCIAN', 'KATAKANA', 'JAVANESE', 'ARABIC', 'KANNADA', 'BUHID',
                    'TAGBANWA', 'DESERET', 'REJANG', 'BOPOMOFO', 'PERMIC', 'OSAGE', 'TAGALOG', 'MEETEI', 'CARIAN', 
                    'UGARITIC', 'ORIYA', 'ELBASAN', 'CYPRIOT', 'HANUNOO', 'GUJARATI', 'LYDIAN', 'MONGOLIAN', 'AVESTAN',
                    'MEROITIC', 'KHAROSHTHI', 'HUNGARIAN', 'KHUDAWADI', 'ETHIOPIC', 'PERSIAN', 'OSMANYA', 'ELBASAN',
                    'TIBETAN', 'BENGALI', 'TURKIC', 'THROWING', 'HANIFI', 'BRAHMI', 'KAITHI', 'LIMBU', 'LAO', 'CHAKMA',
                    'DEVANAGARI', 'ITALIC', 'CJK', 'MEDEFAIDRIN', 'DIAMOND', 'SAURASHTRA', 'ADLAM', 'DUPLOYAN'
                    ]
        self.disallowed_codes = ['1F1A4', 'A7AF']
        
        self.dces_p_file = PerturbationsStorage(args.dces_p_file)
        self.topn = args.dces_topn
        self.prob = args.dces_p
        self.mydict = {}
        self.descs = self.get_descs()
        
    
    def get_descs(self) :
        # load the unicode descriptions into a single dataframe with the chars as indices\
        with open('/home/nykim/HateSpeech/03_code/VIPER/NamesList.txt', 'r') as f :
            count_columns = [len(l.split('\t')) for l in f.readlines()[16:]]
            f.close()
        columns = [i for i in range(0, max(count_columns))]
        columns[:2] = ['code', 'description']
        descs = pd.read_csv('/home/nykim/HateSpeech/03_code/VIPER/NamesList.txt', skiprows=np.arange(16), header=None, names=columns, delimiter='\t')
        descs.drop(columns=[2, 3], inplace=True)
        descs = descs.dropna(axis=0)
        descs_arr = descs.values # remove the rows after the descriptions
        vectorizer = CountVectorizer(max_features=1000)
        desc_vecs = vectorizer.fit_transform(descs_arr[:, 0]).astype(float)
        vecsize = desc_vecs.shape[1]
        self.vec_colnames = np.arange(vecsize)
        desc_vecs = pd.DataFrame(desc_vecs.todense(), index=descs.index, columns=self.vec_colnames)
        descs = pd.concat([descs, desc_vecs], axis=1)
        return descs
    
    def char_to_hex_string(self, ch) :
        return '{:04x}'.format(ord(ch)).upper()
    
    def get_all_variations(self, ch) :
        # get unicode number for c
        c = self.char_to_hex_string(ch)
        
        # problem: latin small characters seem to be missing?
        if np.any(self.descs['code'] == c) :
            description = self.descs['description'][self.descs['code'] == c].values[0]
        else :
            return c, np.array([])
        
        # strip away everything that is generic wording, e.g. all words with > 1 character in
        toks = description.split(' ')
        
        case = 'unknown'
        identifiers = []
        hangul_idfs = [
            'KIYEOK', 'SSANGKIYEOK', 'NIEUN', 'TIKEUT', 'SSANGTIKEUT', 'RIEUL', 'MIEUM', 'PIEUP', 'SSANGPIEUP',
            'SIOS', 'SSANGSIOS', 'IEUNG', 'CIEUC', 'SSANGCIEUC', 'CHIEUCH', 'KHIEUKH', 'THIEUTH', 'PHIEUPH', 'HIEUH',
            'A', 'AE', 'YA', 'YAE', 'EO', 'E', 'YEO', 'YE', 'O', 'WA', 'WAE', 'OE', 'YO', 'U', 'WEO', 'WE', 'WI', 'YU',
            'EU', 'YI', 'I'
        ]
        
        for tok in toks :
            if tok in hangul_idfs :
                identifiers.append(tok)
                
                # for debugging 
                if len(identifiers) > 1 :
                    # print('Found multiple ids: ')
                    # print(identifiers)
                    pass
            
            elif tok == 'SMALL':
                case = 'SMALL'
            elif tok == 'CAPITAL':
                case = 'CAPITAL'
            elif tok == 'CHOSEONG':
                case = 'CHOSEONG'
            elif tok == 'JUNGSEONG' :
                case = 'JUNGSEONG'
            elif tok == 'CIRCLED':
                case = 'CIRCLED'
            elif tok == 'PARENTHESIZED' :
                case = 'PARENTHESIZED'
            elif tok == 'HALFWIDTH' :
                case = 'HALFWIDTH'

        # find matching chars
        matches = []
        for i in identifiers:        
            for idx in self.descs.index:
                desc_toks = self.descs['description'][idx].split(' ')
                if i in desc_toks and not np.any(np.in1d(desc_toks, self.disallowed)) and \
                        not np.any(np.in1d(self.descs['code'][idx], self.disallowed_codes)) and \
                        not int(self.descs['code'][idx], 16) > 30000:

                    # get the first case descriptor in the description
                    desc_toks = np.array(desc_toks)
                    # case_descriptor = desc_toks[ (desc_toks == 'SMALL') | (desc_toks == 'CAPITAL') | (desc_toks == 'CHOSEONG') | (desc_toks == 'JUNGSEONG') | (desc_toks == 'CIRCLED') | (desc_toks == 'PARENTHESIZED') | (desc_toks == 'HALFWIDTH')]
                    case_descriptor = desc_toks[(desc_toks == 'CHOSEONG') | (desc_toks == 'JUNGSEONG') | (desc_toks == 'CIRCLED') | (desc_toks == 'PARENTHESIZED') | (desc_toks == 'HALFWIDTH')]

                    if len(case_descriptor) > 1:
                        case_descriptor = case_descriptor[0]
                    elif len(case_descriptor) == 0:
                        case = 'unknown'

                    if case == 'unknown' or case == case_descriptor:
                        matches.append(idx)
        # check the capitalisation of the chars
        return c, np.array(matches) 
    
    # function for finding the nearest neighbours of a given word
    def get_unicode_desc_nn(self, c, dces_p_file, topn=1):
        # we need to consider only variations of the same letter -- get those first, then apply NN
        c, matches = self.get_all_variations(c)
        
        if not len(matches):
            return [], [] # cannot disturb this one
        
        # get their description vectors
        match_vecs = self.descs[self.vec_colnames].loc[matches]
            
        # find nearest neighbours
        neigh = NearestNeighbors(metric='euclidean')
        Y = match_vecs.values
        neigh.fit(Y) 
        
        X = self.descs[self.vec_colnames].values[self.descs['code'] == c]

        if Y.shape[0] > topn:
            dists, idxs = neigh.kneighbors(X, topn, return_distance=True)
        else:
            dists, idxs = neigh.kneighbors(X, Y.shape[0], return_distance=True)

        # turn distances to some heuristic probabilities
        #print(dists.flatten())
        probs = np.exp(-0.5 * dists.flatten())
        probs = probs / np.sum(probs)
        
        # turn idxs back to chars
        #print(idxs.flatten())
        charcodes = self.descs['code'][matches[idxs.flatten()]]
        
        #print(charcodes.values.flatten())
        
        chars = []
        for charcode in charcodes:
            chars.append(chr(int(charcode, 16)))

        # filter chars to ensure OOV scenario (if perturbations file from prev. perturbation contains any data...)
        c_orig = chr(int(c, 16))
        chars = [char for char in chars if not dces_p_file.observed(c_orig, char)]

        #print(chars)

        return chars, probs
    
    def do_dces(self, sentence) :
        out_x = []
        for c in sentence :
            if c not in self.mydict :
                similar_chars, probs = self.get_unicode_desc_nn(c, self.dces_p_file, topn=self.topn)
                probs = probs[:len(similar_chars)]
                
                # normalise after selecting a subset of similar characters
                probs = probs / np.sum(probs)
                
                self.mydict[c] = (similar_chars, probs)
            
            else :
                similar_chars, probs = self.mydict[c]
                
            r = random.random()
            if r < self.prob and len(similar_chars):
                s = np.random.choice(similar_chars, 1, replace=True, p=probs)[0]
            else:
                s = c
                out_x.append(s)

        return ''.join(out_x)
    
class VIPER_ICES :
    def __init__(self, args) :
        self.w2v = W2Vec.load('/home/nykim/HateSpeech/03_code/VIPER/02_model.bin')
        self.topn = args.ices_topn
        self.prob = args.ices_p
        self.mydict = {}
        self.ices_p_file = PerturbationsStorage(args.ices_p_file)
        
    def do_ices(self, sentence) :
        isOdd, isEven = False, False
        prob = self.prob
        a = sentence.split()
        wwords = []
        out_x = []
        for w in a :
            for c in w :
                if c not in self.mydict :
                    try :
                        similar = self.w2v.most_similar(c, topn=self.topn)
                        if isOdd :
                            similar = [similar[iz] for iz in range(1, len(similar), 2)]
                        elif isEven :
                            similar = [similar[iz] for iz in range(0, len(similar), 2)]
                        words, probs = [x[0] for x in similar], np.array([x[1] for x in similar])
                        probs /= np.sum(probs)
                        self.mydict[c] = (words, probs)
                    except KeyError :
                        s = c
                else :
                    words, probs = self.mydict[c]
                r = random.random()
                if r < prob :
                    s = np.random.choice(words, 1, replace=True, p=probs)[0]
                    self.ices_p_file.add(c, s)
                else :
                    s = c
                out_x.append(s)
            wwords.append("".join(out_x))
            out_x = []
        self.ices_p_file.maybe_write()
        return ' '.join(wwords)