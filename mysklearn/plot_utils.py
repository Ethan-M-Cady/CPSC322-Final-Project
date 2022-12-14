'''
Graph Generating Module for PA3
Contains bar_graph, pie_graph, hist_graph, scatter_graph, and box_plot
'''

from matplotlib import pyplot as plt
import numpy as np

def bar_graph(x_list, y_list, graph_name, axis_label: tuple, user_detail=None) -> None:
    '''A function that generate a bar graph with given inputs

        Args:
        x_list (list): data for the horizontal values.
        y_list (list): data for the vertical values.
        graph_name (str): name of the graph.
        axis_label (tuple): the axis labels for the graph with x and y labels as a tuple.
        user_detail (dict): user inputted graph specifications such as align, width or edgecolor.
    '''
    graph_detail = {"align":"center", "width":0.8, "edgecolor":"black"}
    if user_detail:
        if user_detail["align"]:
            graph_detail["align"] = user_detail["align"]
        if user_detail["width"]:
            graph_detail["width"] = user_detail["width"]
        if user_detail["edgecolor"]:
            graph_detail["edgecolor"] = user_detail["edgecolor"]
    plt.figure(figsize=(20,3))
    plt.title(graph_name)
    plt.bar(x_list, y_list, align=graph_detail["align"], \
        width=graph_detail["width"], edgecolor=graph_detail["edgecolor"])
    if len(x_list) > 4:
        plt.xticks(x_list, x_list, rotation=55, ha="right")
    plt.xlabel(axis_label[0])
    plt.ylabel(axis_label[1])
    plt.show()

def pie_graph(x_list, y_list, graph_name):
    '''A function that generate a pie graph with given inputs

        Args:
        x_list (list): data for the horizontal values.
        y_list (list): data for the vertical values.
        graph_name (str): name of the graph.
    '''
    plt.figure()
    plt.title(graph_name)
    plt.pie(y_list, labels=x_list, autopct="%1.1f%%")
    plt.show()

def hist_graph(data, graph_name):
    '''A function that generate a histogram graph with given inputs

        Args:
        data (list): data for the vertical values.
        graph_name (str): name of the graph.
    '''
    plt.figure()
    plt.title(graph_name)
    plt.hist(data, bins=10)
    plt.show()

def scatter_graph(x_list, y_list, axis_label: tuple, graph_name, slope_intercept: tuple =None):
    '''A function that generate a scatter graph with given inputs

        Args:
        x_list (list): data for the horizontal values.
        y_list (list): data for the vertical values.
        graph_name (str): name of the graph.
        axis_label (tuple): the axis labels for the graph with x and y labels as a tuple.
        slope_intercept (tuple): user inputted slope and intercept for best fit line.
    '''
    plt.figure(figsize=(20,20))
    plt.title(graph_name)
    plt.xlabel(axis_label[0])
    plt.ylabel(axis_label[1])
    plt.scatter(x_list, y_list, marker="x", s=100, c="blue")
    if slope_intercept[0] and slope_intercept[1]:
        plt.plot([min(x_list), max(x_list)], \
            [slope_intercept[0] * min(x_list) + slope_intercept[1], \
                slope_intercept[0] * max(x_list) + slope_intercept[1]], c="red", lw=5)
    plt.tight_layout()
    plt.show()

def box_plot(distributions, labels, axis_label: tuple, graph_name):
    '''A function that generate a box and whisker graph with given inputs

        Args:
        labels (list): data for the horizontal values.
        distributions (list): data for the vertical values.
        graph_name (str): name of the graph.
        axis_label (tuple): the axis labels for the graph with x and y labels as a tuple.
        user_detail (dict): user inputted graph specifications such as align, width or edgecolor.
    '''
    plt.figure(figsize=(20,5))
    plt.title(graph_name)
    plt.xlabel(axis_label[0])
    plt.ylabel(axis_label[1])
    plt.boxplot(distributions)
    plt.xticks(list(range(1, len(distributions) + 1)), labels, rotation=55, ha="right")
    plt.show()

def line_graph(x_list, y_list, axis_label: tuple, graph_name):
    '''A function that generate a scatter graph with given inputs

        Args:
        x_list (list): data for the horizontal values.
        y_list (list): data for the vertical values.
        graph_name (str): name of the graph.
        axis_label (tuple): the axis labels for the graph with x and y labels as a tuple.
        slope_intercept (tuple): user inputted slope and intercept for best fit line.
    '''
    plt.figure(figsize=(20,5))
    plt.title(graph_name)
    plt.xlabel(axis_label[0])
    plt.ylabel(axis_label[1])
    plt.plot(x_list, y_list)
    plt.xticks(x_list, x_list, rotation=55, ha="right")
    plt.tight_layout()
    plt.show()

def multiple_line_graph(x_list, Y_list: dict, axis_label: tuple, graph_name):
    plt.figure(figsize=(20,10))
    for key in Y_list.keys():
        plt.plot(x_list, Y_list[key], label=key)
    plt.axhline(y = 0, color = 'r', linestyle = '-')

    plt.text(28, 3, s="Covid Outbreak", style='oblique', bbox={'facecolor': 'red', 'alpha': 0.75, 'pad': 5})
    plt.text(42, -3, s="Covid Vaccine", style='oblique', bbox={'facecolor': 'green', 'alpha': 0.75, 'pad': 5})
    plt.text(37, 7, s="Covid Vaccine 2nd Dose", style='oblique', bbox={'facecolor': 'green', 'alpha': 0.75, 'pad': 5})
    plt.text(46, 5, s="Delta Variant", style='oblique', bbox={'facecolor': 'red', 'alpha': 0.75, 'pad': 5})
    plt.text(64, -7, s="Omicron Variant", style='oblique', bbox={'facecolor': 'red', 'alpha': 0.75, 'pad': 5})
    
    plt.title(graph_name)
    plt.xlabel(axis_label[0])
    plt.ylabel(axis_label[1])
    
    plt.legend()
    plt.xticks(x_list, x_list, rotation=55, ha="right")
    plt.tight_layout()
    plt.show()

def double_line_graph(x_list, Y_list: dict, axis_label: tuple, graph_name):
    plt.figure(figsize=(20,10))
    for key in Y_list.keys():
        plt.plot(x_list, Y_list[key], label=key)
    plt.axhline(y = 0, color = 'r', linestyle = '-')

    plt.title(graph_name)
    plt.xlabel(axis_label[0])
    plt.ylabel(axis_label[1])
    
    plt.legend()
    plt.xticks(x_list, x_list, rotation=55, ha="right")
    plt.tight_layout()
    plt.show()