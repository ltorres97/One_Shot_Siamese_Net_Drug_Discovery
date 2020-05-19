# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 16:22:05 2019

@author: luist
"""

if __name__=='__main__':
    
  def read_fasta(fp):
        name, seq = None, []
        for line in fp:
            line = line.rstrip()
            if line.startswith(">"):
                if name: yield (name, ''.join(seq))
                name, seq = line, []
            else:
                seq.append(line)
        if name: yield (name, ''.join(seq))


names = []
seqs = []

with open('drug sequences.fasta') as fp:
    for name, seq in read_fasta(fp):
        names.append(name)
        names.append(seq)
        print(name, seq)
        
        