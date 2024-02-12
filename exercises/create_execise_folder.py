import os, time

#########################################################################
#                                                                       #
#    Inputs below this box                                              #
#                                                                       #
#########################################################################

exercise_number = "04" 
number_of_tasks = 5
number_subtasks = []

#########################################################################
#                                                                       #
#   this skript creates templates for the exercises in the same         #
#   folder where the skript is                                          #
#                                                                       #
#   each exercise gets an own folder, each task of an exercise          #
#   gets a own file                                                     #
#                                                                       #
#   input exercise number as String in form "XX", eg "02"               #
#   input number of tasks as simple int                                 #
#   optional give a array containing the number of subtasks             # 
#   for each task                                                       #
#                                                                       #
#   if subtaks are given len(number_subtasks) has to be eqaul           #
#   to number_of_taks                                                   #
#   the default is to asume 3 subtaks per task                          #
#                                                                       #
#########################################################################
#########################################################################
#                                                                       #
#    No changes below here requirred                                    #
#                                                                       #
#########################################################################

start_time = time.time()

folder_path = os.path.join(os.path.dirname(__file__) + "\exercise_" + exercise_number)

if len(number_subtasks) == 0:
    print("\nNo subtasks where given, asume 3 subtasks for each Task.")
    number_subtasks = [3]*number_of_tasks
elif len(number_subtasks) != number_of_tasks:
    print("\nLen of array with subtasks has to equal to number of tasks.")
    exit()

if os.path.exists(folder_path):
    print("\nFolder with the name {} allready exists.".format("\exercise_" + exercise_number))
    exit()
else:
    os.mkdir(folder_path)
    print("\nCreated folder ", folder_path)  

for task in range(1, number_of_tasks+1):
    if task < 10:
        filename = "task_0" + str(task) + ".py"
    else:
        filename = "task_" + str(task) + ".py"

    with open(folder_path + "\\" + filename, 'w') as file:
        file.write("import numpy as np\n")
        file.write("import matplotlib.pyplot as plt\n")
        file.write("import matplotlib.colors as mcolors\n")
        file.write("import cv2 as cv\n")
        file.write("import skimage\n")
        file.write("import time\n")
        file.write("from PIL import Image \n")
        file.write("from scipy import ndimage\n")

        for subtask in range(1, number_subtasks[task-1]+1):
            file.write("\n\n")
            file.write("\n#########################################################################")
            #file.write("\n# Task " + str(task) + str(chr(subtask+96)))
            file.write("\n#\t\t\tExercise "+ exercise_number + "\t\t\t" + "Task " + str(task) + str(chr(subtask+96)) + 9*"\t" + "#")
            file.write("\n#########################################################################")
            file.write("\nprint(\"\\nTask " + str(task) + str(chr(subtask+96)) + "\")")
            file.write("\nstart_time = time.time()")
            file.write("\n")
            file.write("\n# code here")
            file.write("\n")
            file.write("\nend_time = time.time()")
            file.write("\nprint(\"Completetd in {}s.\".format(end_time-start_time))")

    print("\nCreated file {} with {} subtasks".format(filename, number_subtasks[task-1]))

end_time = time.time()
print("Completetd in {}s.".format(end_time-start_time))
