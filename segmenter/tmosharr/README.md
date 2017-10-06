> This folder contains all files developed and/or used by `tmosharr` in order to solve the **Word Segmentation - Homework1**

Check the [`README.lperesde`](../answer/README.tmosharr) for information about the approach used on the project: tasks distribution, algorithms and such.

##### Table of contents:
  - [./baseline.py](#baselinepy)
  - [./bigram.py](#bigram.py)
  - [./data/](#data)
  - [./data_wseg/](#data_wseg)
    - count_wseg.txt
    - count_wseg_1w.txt
    - prepare_wseg_cn.py
  - [./history](history)
  - [./improve_baseline.py](#improve_baselinepy)
  - [./scores](#scores)
  - [./study_of_case.py](#study_of_case.py)
  - [./readme.tex](#readmetex)

###### baseline.py
The baseline algorithm given for this project.

###### bigram.py
Implementation of baseline, this time checking unigrams and bigrams.

###### data
Data given by staff for project baseline.

###### data_wseg
Folder with data parsed from `wseg_simplified_cn` (1 million chinese words) file.

In this folder, `prepare_wseg_cn.py` script was used to parse `wseg_simplified_cn` into `count_wseg.txt` (same format as count_1w.txt). After that, we used its output and merged the files `count_1w.txt` and `count_wseg.txt`, generating a very powerful unigram data file: `count_wseg_1w.txt`

###### history
Output of commits using `git log`.

###### improve_baseline.py
 Script that executes parsing and processing of chinese characters based on data pattern such as numbers, date and unknown words. This is the main file in this folder, once it uses the file [./wseg_data/count_wseg_1w.txt](wseg_data/count_wseg_1w.txt) (This file is an union of the files [./data/count_1w.txt](data/count_1w.txt) and [./data_wseg/count_wseg.txt](data/count_wseg.txt), intending to maximize the score of our solution).

###### scores
Gathering of scores from command `python improve_baseline.py | python score-segments.py`

###### study_of_case.py
First file used on the segmenter project (it was not used).

###### readme.tex
A first LaTeX README version for the algorithm and tasks developed in this project
