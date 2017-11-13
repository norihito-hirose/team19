import os
import re

# get hoge.py from data_dir
def get_files(data_dir):
    target_files = []
    for root, dirs, files in os.walk(data_dir):
        targets = [os.path.join(root, f) for f in files if f.endswith(".py")]
        target_files.extend(targets)
    return target_files

# extracet tf.hoge from code
def find_tf(file_path):
    f = open(file_path, "r")
    code = f.read()
    elements = re.findall("tf\..*?\(", code)
    elements = [element.replace("(", "") for element in elements]

    return elements

data_dir = os.path.join(os.getcwd(), "data/raw")
files_path = get_files(data_dir)

if not os.path.exists("data/input"):
    os.makedirs("data/input")

count = 0

for file_path in files_path:
    try:
        calls = find_tf(file_path)
    except:
        continue
    if len(calls) < 2:
        continue
    save_path = os.path.join(os.getcwd(), "data/input", "code%s.txt" % count)
    f = open(save_path, "a")
    f.writelines("\n".join(calls))
    f.close()
    count += 1
