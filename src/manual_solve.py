#!/usr/bin/python

# Student Information
# Student Name: Manmaya Prasad Panda 
# Student ID: 17232977
# Git: https://github.com/manmayajob/ARC

import os, sys
import json
import numpy as np
import re
import matplotlib.pyplot as plt

### YOUR CODE HERE: write at least three functions which solve
### specific tasks by transforming the input x and returning the
### result. Name them according to the task ID as in the three
### examples below. Delete the three examples. The tasks you choose
### must be in the data/training directory, not data/evaluation.




def solve_6150a2bd(inputs):
    '''
    In this function, numpy package to reverse the order of elements in an array, 
    the elements are reordered but the shape is preserved.
    
    Parameters: c = ARC test list
    
    Returns: tst_rslt = returns a numpy ndarray, size of which depends on the task

    '''

    input_array = np.array(inputs)
    tst_rslt = np.flip(input_array)
    return tst_rslt

def solve_ce22a75a(inputs):
    '''
    The function returns a numpy array by replacing all zeros around a pattern with the corresponding pattern value.
    
    '''
    tst_rslt = np.array(inputs)
    res = np.where(tst_rslt > 0)
    try:
        
        for i in list(zip(res[0], res[1])):     
            for x in range(-1, 2):
                for y in range(-1, 2):
                    if (x != 0 or y != 0):
                        tst_rslt[i[0] - x][i[1] - y] = 1
                        tst_rslt[i[0]][i[1]] = 1 
    except IndexError: # catch the error
        pass #we are passing the if any point is at the corner                         
    return tst_rslt

def solve_1cf80156(inputs):
    '''

    The function returns an numpy array by slicing the min and max value of rows and columns for the given input.
    
    '''

    input_array = np.array(inputs)
    result = np.where(input_array > 0)
    column_minium = min(result[1])
    column_maximum = max(result[1])
    row_minimum = min(result[0])
    row_maximum = max(result[0])
    tst_rslt = []
    for i in range(row_minimum,row_maximum + 1):
        for j in range(column_minium,column_maximum + 1):
            tst_rslt.append(input_array[i][j])
    tst_rslt = np.reshape(tst_rslt,(row_maximum-row_minimum + 1,column_maximum-column_minium + 1))
    return tst_rslt

def main():
    # Find all the functions defined in this file whose names are
    # like solve_abcd1234(), and run them.

    # regex to match solve_* functions and extract task IDs
    p = r"solve_([a-f0-9]{8})" 
    tasks_solvers = []
    # globals() gives a dict containing all global names (variables
    # and functions), as name: value pairs.
    for name in globals(): 
        m = re.match(p, name)
        if m:
            # if the name fits the pattern eg solve_abcd1234
            ID = m.group(1) # just the task ID
            solve_fn = globals()[name] # the fn itself
            tasks_solvers.append((ID, solve_fn))

    for ID, solve_fn in tasks_solvers:
        # for each task, read the data and call test()
        directory = os.path.join("..", "data", "training")
        json_filename = os.path.join(directory, ID + ".json")
        data = read_ARC_JSON(json_filename)
        test(ID, solve_fn, data)
    
def read_ARC_JSON(filepath):
    """Given a filepath, read in the ARC task data which is in JSON
    format. Extract the train/test input/output pairs of
    grids. Convert each grid to np.array and return train_input,
    train_output, test_input, test_output."""
    
    # Open the JSON file and load it 
    data = json.load(open(filepath))

    # Extract the train/test input/output grids. Each grid will be a
    # list of lists of ints. We convert to Numpy.
    train_input = [np.array(data['train'][i]['input']) for i in range(len(data['train']))]
    train_output = [np.array(data['train'][i]['output']) for i in range(len(data['train']))]
    test_input = [np.array(data['test'][i]['input']) for i in range(len(data['test']))]
    test_output = [np.array(data['test'][i]['output']) for i in range(len(data['test']))]

    return (train_input, train_output, test_input, test_output)


def test(taskID, solve, data):
    """Given a task ID, call the given solve() function on every
    example in the task data."""
    print(taskID)
    train_input, train_output, test_input, test_output = data
    print("Training grids")
    for x, y in zip(train_input, train_output):
        yhat = solve(x)
        show_result(x, y, yhat)
    print("Test grids")
    for x, y in zip(test_input, test_output):
        yhat = solve(x)
        show_result(x, y, yhat)
    plt_list = train_input + test_input 
    visualization_func(plt_list,taskID)  # added to show the grid visualisation of the results 
        
def show_result(x, y, yhat):
    print("Input")
    print(x)
    print("Correct output")
    print(y)
    print("Our output")
    print(yhat)
    print("Correct?")
    if y.shape != yhat.shape:
        print(f"False. Incorrect shape: {y.shape} v {yhat.shape}")
    else:
        print(np.all(y == yhat))

def visualization_func(input,titlename):
    '''
    plotting grids and emulating testing interface.
    
    Parameters: input = A list of test input and computed output.

    '''
    for i in range(len(input)):
        plt.matshow(input[i])

    plt.title(titlename)    
    plt.show()


if __name__ == "__main__": main()

