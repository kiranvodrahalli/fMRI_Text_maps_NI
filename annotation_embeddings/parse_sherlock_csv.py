#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import pickle as p
from math import floor

path = '/Users/kiranv/home/thesis/sherlock/'
oov = 'oov_sherlock.txt'


# dict for what to swap with what
def swapdict_oov(oov):
	d = dict()
	with open(oov, 'rb') as f:
		for line in f:
			o, t = line.split("*")[0], line.split("*")[1]
			if t.strip(" \n") == 'xxxx':
				t = ''
			else:
				t = t.strip(" \n")
			o = o.strip(" \n")
			d[o] = t
	return d

start_time = 4 # col 
end_time = 5 # col
text = 6 # col

lines = []
with open(path + 'Sherlock_Segments_master032716.csv', 'rb') as csvfile:
	reader = csv.reader(csvfile, delimiter=',', quotechar='"')
	cnt = 0
	col_names = []
	swapdict = swapdict_oov(path + oov)
	print swapdict
	for row in reader:
		if cnt == 0:
			col_names = row
		elif cnt == 1001:
			continue
		else:
			txt = row[text]
			txt = txt.replace('…', ' ')
			txt = txt.replace("'", '')
			txt = txt.replace('"', ' ')
			txt = txt.replace('.', ' ')
			txt = txt.replace(',', ' ')
			txt = txt.replace('?', ' ')
			txt = txt.replace('!', ' ')
			txt = txt.replace(':', ' ')
			txt = txt.replace(';', ' ')
			txt = txt.replace('*', ' ')
			txt = txt.replace('(', ' ')
			txt = txt.replace(')', ' ')
			txt = txt.replace('[', ' ')
			txt = txt.replace(']', ' ')
			txt = txt.replace('/', ' ')
			txt = txt.replace('\\', ' ')
			txt = txt.replace('&', ' ')
			txt = txt.replace('<', ' ')
			txt = txt.replace('>', ' ')
			txt = txt.replace('-', ' ')
			txt = txt.replace('’', '')
			txt = txt.replace('\xa0', ' ')
			txt = txt.replace('\xc2', ' ')
			txt = txt.replace('0', ' zero ')
			txt = txt.replace('9', ' nine ')
			txt = txt.replace('8', ' eight ')
			txt = txt.replace('7', ' seven ')
			txt = txt.replace('6', ' six ')
			txt = txt.replace('5', ' five ')
			txt = txt.replace('4' , ' four ')
			txt = txt.replace('3', ' three ')
			txt = txt.replace('2', ' two ')
			txt = txt.replace('1', ' one ')
			txt = txt.lower()
			new_txt = ''
			for s in txt.split(" "):
				s = s.strip(" \n")
				if s in swapdict:
					s = swapdict[s]
				new_txt += s + " "
			txt = new_txt.split(" ")
			new_txt = []
			for t in txt:
				if (t != '' and t != ' '):
					new_txt.append(t)
			txt = new_txt
			t1 = row[start_time]
			t2 = row[end_time]
			t1 = float(t1)
			t2 = float(t2)
			lines.append((t1, t2, txt))
		cnt += 1

print txt
old_last = lines[len(lines) - 1]
lines[len(lines) - 1] = (old_last[0], 1544, old_last[2]) #last 2 TRs have no fMRI data: 1544-1545.5, 1545.5 - 1547
del lines[481] # no fMRI data; just a black screen: two "segments", each of length 3 seconds

p.dump(lines, open('sherlock_text_times.p', 'wb'))

# match text with TRs
TR = 1.5
TRs = [[] for i in range(1976)]
curr_TR = 0
total_secs = 0.0
rem = 0.0 # remainder
prev_t2 = -1
lnum = 2
for l in lines:
	t1, t2, txt = l[0], l[1], l[2]
	if prev_t2 == -1:
		prev_t2 = t2
	if t1 != prev_t2:
		print lnum
		print t1
		print t2
		print prev_t2
		print "=========="
	prev_t2 = t2;
	lnum += 1
	diff = rem + t2 - t1
	nTRs = floor(diff/TR)
	nTRs = int(nTRs)
	rem = diff - nTRs*TR
	#print str(TR) + " * " + str(nTRs) + " = " + str(nTRs*TR) + "; Rem: " + str(rem)
	total_secs += nTRs*TR 
	#print "Total seconds: " + str(total_secs)

with open('sherlock_text.txt', 'wb') as f:
	for l in lines:
		print curr_TR
		t1, t2, txt = l[0], l[1], l[2]
		diff = rem + t2 - t1
		nTRs = floor(diff/TR)
		nTRs = int(nTRs)
		rem = diff - nTRs*TR
		for i in range(nTRs):
			#print curr_TR + i
			TRs[curr_TR + i].append(txt)
		if nTRs > 0:
			curr_TR += (nTRs -1)
		if rem > 0:
			TRs[curr_TR + 1].append(txt)
			#print curr_TR + 1
		if nTRs > 0:
			curr_TR += 1 # move to the next TR

		## creating the text part
		for w in txt:
			f.write(w + " ")
		f.write("\n")

new_TRs = []
for tr in TRs:
	new_TRs.append(tr)
#for tr in TRs:
#	new_tr = ''
#	for s in tr:
#		new_tr += s
#	new_s = s.split(" ")
#	new_new_s = []
#	for x in new_s:
#		x = x.strip("\n")
#		if x != '':
#			new_new_s.append(x)
#	#print new_new_s
#	new_TRs.append(new_new_s)
TRs = new_TRs
print len(TRs)
p.dump(TRs, open('sherlock_text_TRs.p', 'wb'))

# DO NEW ANALYSIS OF THE TRS
# add interactive addition from the small atom set which I pre-selected!
# make sure to build a dict of relevant words based on top 5
# also add the characters in scene/location/who's speaking/name-focus













