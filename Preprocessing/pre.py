###############################################################
# A small preprocessing block that split lines by \n character
#################################################################

textfile = open("Output.txt").read()
new=""
for i in xrange(0,len(textfile)):
	new += textfile[i]
	if textfile[i] == ".":
		new+="\n"

file = open("OutputNew.txt","w")
file.write(new)
file.close()