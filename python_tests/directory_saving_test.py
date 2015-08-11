import os, sys

# use python comprehension to generate list of files in directory
# directories = [name for name in os.listdir(".") if os.path.isdir(name)]

# create a new unique ID

def check_create_directory(new_folder):

    # finds the current path, and creates new data directory
    path = os.path.dirname(os.path.realpath(__file__))
    data_directory = path + "/" + new_folder

    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

def new_dir_index(sub_folder):

  # get list of all directories
      # finds the current path, and creates new data directory
  path = os.path.dirname(os.path.realpath(__file__))
  data_directory = path + "/" + sub_folder

  all_dirs = [name for name in os.listdir(data_directory) ] # if os.path.isdir(name)

  # split each directory "ex-N" to get highest N
  last_experiment = 0
  print len(all_dirs)

  if len(all_dirs) > 0:

    print ">0"
    for i in range(len(all_dirs)):
      # assuming folder is 'ex-N'
      front, dash, end = all_dirs[i].rpartition("-")
      if len(front) > 0:
        end = int(end)
        if end > last_experiment:
          last_experiment = end

    # return new experiment number
    return last_experiment + 1

  # otherwise 
  return 1

def new_experiment_folder():

  # check in subfolder
  subfolder = "test"

  # get next experiment number
  num = new_dir_index(subfolder)

  # new folder name
  foldername = "{}/ex-{}".format(subfolder,num)

  # check if that folder exists, if not create it
  check_create_directory(foldername)

  print foldername

