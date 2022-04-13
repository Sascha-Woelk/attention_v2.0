# import libraries
from modules.setup_file import *

# find filenames of confusion matrices
files = glob.glob('files/confusion_matrices/class*.pickle')

# extract unique class numbers from filenames
classes = []
for i in range(len(files)):
    class_num = int(files[i].split('class')[1].split('_')[0])
    classes.append(class_num)
run_classes = np.unique(classes)

# write run_classes to a pickle file
with open('files/run_classes.pickle', 'wb') as file:
    pickle.dump(run_classes, file)
