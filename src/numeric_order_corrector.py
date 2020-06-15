import os
import natsort


def rearrange_numeration():
	folder_path = "../dataset/lines/genesis/000"
	names = os.listdir(folder_path)

	left_names = [name for name in names if "left" in name]
	left_names = natsort.natsorted(left_names)
	right_names = [name for name in names if "right" in name]
	right_names = natsort.natsorted(right_names)

	index = 0

	for name in left_names:
		os.rename(os.path.join(folder_path, name), os.path.join(folder_path, "left_line_{}.png".format(index)))
		index += 1

	index = 0

	for name in right_names:
		os.rename(os.path.join(folder_path, name), os.path.join(folder_path, "right_line_{}.png".format(index)))
		index += 1