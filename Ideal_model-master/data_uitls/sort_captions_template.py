# -*- coding = utf-8 -*-

caption_dict={}
with open('/Users/apple/Desktop/Ideal_model/intermediate_files/template_captions.txt') as f:
    caption_list=f.readlines()
    
for c in caption_list:
    i = c.index(':')
    caption_dict[int(c[:i-5])] = c[i+2:]
sorted(caption_dict.keys())

with open('sorted_template_captions.txt','w+') as wf:
    for c in caption_dict:
        string = str(c).zfill(6)+'.jpg'+' : ' + caption_dict[c]
        wf.write(string)
        
