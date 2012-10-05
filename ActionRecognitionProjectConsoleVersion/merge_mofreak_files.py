import os
from collections import defaultdict
mofreak_root = "C:/data/TRECVID/mosift/20071107_full2_mofreak/"
svm_root = "C:/data/TRECVID/svm/"
bow_root = "C:/data/TRECVID/bow/"
action = "TestSequence"

def combineBOW():
	out = open(svm_root + "testing.test", 'w')
	files_written = 0
	files = os.listdir(bow_root)



	parsed_names = []
	for file in files:
		parsed_name = file.split(".")
		parsed_names.append(parsed_name)

	id_1 = 1

	id_1_matches = []
	while True:
		# gather all files with first-level-id matches
		found_id_1_match = False
		for name in parsed_names:
			if int(name[-4]) == id_1:
				id_1_matches.append(name)
				found_id_1_match = True

		if not found_id_1_match:
			break
		else:
			id_1 += 1



		id_2 = 0
		while True:
			# gather second-level-id matches from first-level-id matches
			found_id_2_match = False

			for name in id_1_matches:
				if int(name[-3]) == id_2:
					# write this one to the output file.
					inp = open(bow_root + "/" + ".".join(name), 'r')
					lines = inp.readlines()
					print "writing a file...", files_written
					print ".".join(name)
					print id_2
					files_written += 1
					for line in lines:
						out.write(line)

					found_id_2_match = True
					id_2 += 1
					print id_2
					break

			if not found_id_2_match:
				break

		#clear the id1 matches.
		del id_1_matches[0:len(id_1_matches)]



	out.close()

# i accidentally overwrote some of htis code so it might not work anymore
def combine():
	out = open(svm_root + "training.train", 'w')

	files = os.listdir(svm_root)
	for file in files:



		inp = open(svm_root + file, 'r')
		lines = inp.readlines()
		for line in lines:
			out.write(line)

		inp.close()

	out.close()


def chop():
	files = os.listdir(mofreak_root)

	for file in files:
		if not os.path.isdir(mofreak_root + file):
			f = open(mofreak_root + file, 'r')
			lines = f.readlines()

			# chop it into 40k lined chunks.
			if len(lines) > 40000:
				name_parts = file.split(".")

				out = open("tmp", 'w')
				file_index = 0
				for i in xrange(len(lines)):
					if i % 40000 == 0:
						out.close()
						new_name = ".".join(name_parts[:-1]) + "." + str(file_index) + ".mofreak"
						out = open(mofreak_root + "/chop/" + new_name, 'w')
						file_index += 1

					out.write(lines[i])

				out.close()


def merge():
	file_grouping = defaultdict(list)
	files = os.listdir(mofreak_root)

	for file in files:
		# parse the filename.
		if file.strip() ==  "test":
			continue
		else:
			print file

		parts = file.split(".")
		file_grouping[parts[0]].append(file)

	# parse each group to make one file per video.
	for group in file_grouping.keys():
		print "processing ", group
		outp = open(mofreak_root + "/test/" + group + ".mpeg." + action + ".mofreak", "w")

		# parse the filenames so we can sort by their id
		parsed_filenames = []
		for mofreak_file in file_grouping[group]:
			parsed_filenames.append(mofreak_file.split("."))

		file_id = 1#0
		while True:
			found = False
			print "processing id ", file_id

			# do stuff
			for parsed in parsed_filenames:
				if int(parsed[-2]) == file_id:
					file_to_append = ".".join(parsed)

					inp = open(mofreak_root + file_to_append, "r")
					lines = inp.readlines()
					for line in lines:
						outp.write(line)

					found = True
					break

			if found:
				file_id += 1
			else:
				break

if __name__ == '__main__':
	#merge()
	#chop()
	#combine()
	combineBOW()
	print "all done?"