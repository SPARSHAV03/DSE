import os
print("Operating system:", os.name)
print("Working directory:",os.getcwd())
files_and_dirs=os.listdir()
files=[]
directories=[]
for item in files_and_dirs:
  if os.path.isfile(item):
    files.append(item)
  elif os.path.isdir(item):
    directories.append(item)

  print("\n Files:")
  for file in files:
    print(file)
  print("\n Directories:")
  for directory in directories:
    print(directory)