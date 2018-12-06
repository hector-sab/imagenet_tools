import json
#dmap = {'id':['class_name',int_val]}
dmap = {}

file = 'readable_id_classes.txt'
json_file = 'id2class.json'
with open(file,'r') as f:
	lines = f.readlines()

for i,line in enumerate(lines):
	line = line.strip('\n')
	line = line.split(' ')
	dmap[line[0]] = [line[-1],i]

with open(json_file,'w') as f:
	f.write(json.dumps(dmap,indent=4)) # ,sort_keys=True))