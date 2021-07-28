# -*- coding: utf-8 -*-
"""
Author: Irem Ecem Yelkanat
Written with Python v3.8.2
"""

import os

if __name__ == "__main__":

    message = "\nHere you can choose a question to execute\nEnter 1 to execute Question #1\nEnter 2 to execute Question #2\nEnter 3 to execute Question #3\nEnter Q to exit: "
    selection = input(message)
    while selection != "q" and selection != "Q":
        if selection == "1":
            os.system('python3 question_1.py')
        elif selection == "2":
            os.system('python3 question_2.py')
        elif selection == "3":
            os.system('python3 question_3.py')
        else:
            print("Invalid Selection!")
        selection = input(message)
        