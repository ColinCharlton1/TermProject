# TermProject
System: Windows 10

Quick Start Section:
First, install NetLogo version 6.1.0. no version after that works with PyNetLogo
- https://ccl.northwestern.edu/netlogo/download.shtml

Next, setup python and tensorflow, I used Python version 3.8.5 and tensorflow 2.4.1
The easiest way to do this is to follow the guide here:
-https://www.tensorflow.org/install/pip#windows

After that, use pip to install the latest versions of the following libraries
- Numpy: https://numpy.org/install/
- numba: https://numba.pydata.org/numba-doc/latest/user/installing.html
- pandas: https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html
- PyNetLogo: https://pynetlogo.readthedocs.io/en/latest/install.html

After all thats succesfully done, open up ConfigurationManager.py your text editor of choice. 
It contains all the variables you can edit to modify runs of the code.
I recomend taking a close look at the variables, I tried to comment well on all the ones that could be confusing.
Most are hopefully named in an easy to understand way.
The settings as is are fairly good for a basic run. 
Some of the settings affect memory heavily:
- The replay memory of each actor is stored in RAM, so any setting which modifies that has an upper limit based on your available memory
- Each Island is a seperate process running a headless instance of NetLogo, these each take up to 500MB of memory

The number of processes acting at any one time can be controlled by the NUM_PROCESSES setting to prevent it from using too many cores

Once the ConfigurationManager looks good to you, follow these steps: 
- open a command prompt
- set the Current Working Directory (CWD) to the file named 'main'
- run the command: python main.py
- a NetLogo GUI should open up and start displaying island 0 as it progresses
- data will be stored if you feel like taking a look at it afterwards
- models will also be stored every 5 generations

If after training you want to have a showcase of how the models performed every 5 generations, then:
- don't alter any config settings
- set the generations you want to view in the variables located in the showcase.py file
- while still in CWD, run: python showcase.py

Code which was adapted or taken from others code is labelled with comments in the files
