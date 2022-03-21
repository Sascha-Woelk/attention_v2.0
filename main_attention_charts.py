# import libraries
from matplotlib.pyplot import cla
from modules.setup_file import *
import glob
import pandas as pd

import seaborn as sns
sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})
sns.set_style('darkgrid')

from scipy.stats import norm
import math

# identify current working directory and set up subdirectories
working_directory = os.getcwd()
charts_dir = os.path.join(working_directory, 'charts/')

# Define function to calculate d-prime measures
Z = norm.ppf
def SDT(hits, misses, fas, crs):
    """ returns a dict with d-prime measures given hits, misses, false alarms, and correct rejections"""
    # Floors an ceilings are replaced by half hits and half FA's
    half_hit = 0.5 / (hits + misses)
    half_fa = 0.5 / (fas + crs)
 
    # Calculate hit_rate and avoid d' infinity
    hit_rate = hits / (hits + misses)
    if hit_rate == 1: 
        hit_rate = 1 - half_hit
    if hit_rate == 0: 
        hit_rate = half_hit
 
    # Calculate false alarm rate and avoid d' infinity
    fa_rate = fas / (fas + crs)
    if fa_rate == 1: 
        fa_rate = 1 - half_fa
    if fa_rate == 0: 
        fa_rate = half_fa
 
    # Return d', beta, c and Ad'
    out = {}
    out['d'] = Z(hit_rate) - Z(fa_rate)
    out['beta'] = math.exp((Z(fa_rate)**2 - Z(hit_rate)**2) / 2)
    out['c'] = -(Z(hit_rate) + Z(fa_rate)) / 2
    out['Ad'] = norm.cdf(out['d'] / math.sqrt(2))
    
    return(out)

# create empty dataframe to hold summary signal detection metrics per class x intensity
data = pd.DataFrame(columns=['class',
                             'intensity',
                             'hit_rate',
                             'false_alarm_rate',
                             'd_prime',
                             'criterion'])

# find filenames of confusion matrices
files = glob.glob('files/confusion_matrices/class*.pickle')
  
# loop over files
for i in range(len(files)):
  filename = str(files[i])
  class_number = int(filename[filename.find('class')+len('class'):filename.rfind('_intensity')])
  intensity_level = float(filename[filename.find('intensity')+len('intensity'):filename.rfind('.pickle')])/1000
  
  with open(files[i], 'rb') as file:
    confusion_matrix = pickle.load(file)
    
  # calculate signal detection metrics
  stimuli_per_class = confusion_matrix.numpy().sum(axis=1) 
  predictions_per_class = confusion_matrix.numpy().sum(axis=0)

  true_positives = np.diag(confusion_matrix)
  false_positives = predictions_per_class - true_positives
  false_negatives = stimuli_per_class - true_positives
  true_negatives = confusion_matrix.numpy().sum() - (true_positives + false_positives + false_negatives)

  hit_rate = true_positives/(true_positives + false_negatives)
  false_alarm_rate = false_positives/(false_positives + true_negatives)
  
  # select metrics for the target class
  class_true_positives = true_positives[class_number]
  class_false_positives = false_positives[class_number]
  class_false_negatives = false_negatives[class_number]
  class_true_negatives = true_negatives[class_number]
  
  class_hit_rate = hit_rate[class_number]
  class_false_alarm_rate = false_alarm_rate[class_number]
  
  # calculate d-prime metrics for the target class
  signal_metrics = SDT(class_true_positives,
                       class_false_negatives,
                       class_false_positives,
                       class_true_negatives)
  class_d_prime = signal_metrics['d']
  class_criterion =signal_metrics['c']
  
  # add all metrics to summary dataframe
  new_row = {'class': class_number,
             'intensity': intensity_level,
             'hit_rate': class_hit_rate,
             'false_alarm_rate': class_false_alarm_rate,
             'd_prime': class_d_prime,
             'criterion': class_criterion}
  
  data = data.append(new_row, ignore_index=True)

# create charts
measures = ['hit_rate', 'false_alarm_rate', 'd_prime', 'criterion']

for measure in measures:
  g = sns.catplot(x="intensity", y=measure, kind="violin", inner=None, data=data)
  if measure in ['hit_rate', 'false_alarm_rate']:
    g.set(ylim=(0, 1))
  sns.swarmplot(x="intensity", y=measure, color='black', size=3, data=data, ax=g.ax)
  sns.pointplot(x="intensity", y=measure, markers='_', color='red', linestyles='', data=data)
  plt.title(measure)
  plt.xlabel('attention intensity')
  plt.savefig(charts_dir + 'attention_levels/{}{}.png'.format(measure, dt.datetime.today().date()), dpi=300, bbox_inches='tight')