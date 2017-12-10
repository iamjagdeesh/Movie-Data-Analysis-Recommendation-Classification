Vector space modeling of MovieLens & IMDB data - Phase 2

Hardware requirements:
Operating System: Windows 10 Home
Processor: Intel� Core� i5-6200U CPU @ 2.30Ghz 2.40Ghz
System type: 64-bit operating system, X64 based processor

Software requirements:
1. Python 3.6.2 :: Anaconda, Inc.
2. Install gensim for LDA. Run the below command in anaconda command prompt
conda install -c anaconda gensim
Run anaconda prompt as administrator if you encounter permission issues
3. Install tensorly package for CP decomposition.

Directory Structure:
The project directory structure has the following directories:
	1. "resources" - contains the csv files that constitute the data set. 
	2. "scripts" - contains command line interface along with the other supporting scripts needed for the successful execution of the project.

Execution Steps:
Help: This describes how to use the command line interface
Usage: python <command-line-interface> --help
Example: python phase_2_task_4.py --help

Task 1a:
Command line interface - phase_2_task_1a.py
Usage: python phase_2_task_1a.py <genre> <model>
Example: python phase_2_task_1a.py Action svd

Task 1b:
Command line interface - phase_2_task_1b.py
Usage: python phase_2_task_1b.py <genre> <model>
Example: python phase_2_task_1b.py Action pca

Task 1c:
Command line interface - phase_2_task_1c.py
Usage: python phase_2_task_1c.py <actor-id> <model>
Example: python phase_2_task_1c.py 123 tfidf

Task 1d:
Command line interface - phase_2_task_1d.py
Usage: python phase_2_task_1d.py <movie-name> <model>
Example: python phase_2_task_1d.py Daredevil lda

Task 2a:
Command line interface - phase_2_task_2a.py
Usage: python phase_2_task_2a.py
Example: python phase_2_task_2a.py

Task 2b:
Command line interface - phase_2_task_2b.py
Usage: python phase_2_task_2b.py
Example: python phase_2_task_2b.py

Task 2c:
Command line interface - phase_2_task_2c.py
Usage: python phase_2_task_2c.py
Example: python phase_2_task_2c.py

Task 2d:
Command line interface - phase_2_task_2d.py
Usage: python phase_2_task_2d.py
Example: python phase_2_task_2d.py

Task 3:
Command line interface - phase_2_task_3.py
Usage: python phase_2_task_3.py <similarity-matrix-type> <comma-separated-seed-actors>
Example: python phase_2_task_3.py actor 3619702,3426176

Task 4:
Command line interface - phase_2_task_4.py
Usage: phase_2_task_4.py <user-id>
Example: python phase_2_task_4.py 3

Troubleshooting:
	1. The tag weight calculations are performed dynamically whenever the input is passed to the command line interface. Also, this project uses in-memory data frames (via python pandas library) for storage and retrieval. You may observe delay in the output of the command line interface based on the input. Please be patient.
	2. Please ensure the data set (csv files) have the same names and column descriptors as the sample data set for correct execution.
	3. Ensure you are running the correct python interpreter. The correct interpreter will give the following output on the command line:
	 python --version
	 Python 3.6.2 :: Anaconda, Inc.
