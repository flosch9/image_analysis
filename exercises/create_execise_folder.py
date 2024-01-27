import os

#########################################################################
#                                                                       #
#    Inputs below this box                                              #
#                                                                       #
#########################################################################

exercise_number = "03" 
number_of_tasks = 6
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

folder_path = os.path.join(os.path.dirname(__file__) + "\exercise_" + exercise_number)


os.mkdir(os.path.join(os.path.dirname(__file__) + "\exercise_" + exercise_number))   


if len(number_subtasks) == 0:
    print("No subtasks where given.")


for task in range(1, number_of_tasks+1):
    if task < 10:
        filename = "task_0" + str(task) + ".py"
    else:
        filename = "task_" + str(task) + ".py"

    with open(folder_path + "\\" + filename, 'w') as file:
        file.write("import numpy as np\n")
        file.write("import matplotlib.pyplot as plt\n")
        file.write("import matplotlib.pyplot as plt\n")
        file.write("import matplotlib.pyplot as plt\n")
        file.write("import matplotlib.pyplot as plt\n")
        file.write("import matplotlib.pyplot as plt\n")
        file.write("import matplotlib.pyplot as plt\n")
        file.write("import matplotlib.pyplot as plt\n")
        




