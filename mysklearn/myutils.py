import numpy as np

def group_by(table: list, group_by_col_index):
    '''return a list of unique names and subtables from the original
        table that contains unique name.

        Args:
            table (list): the data source for grouping.
            group_by_col_name: the name of the columns for the table to be grouped by.
        Returns:
            group_names (list): A list of each unique names
            group_subtables (list): a list of tables that contains the unique names.
    '''
    labels = [row[group_by_col_index] for row in table]
    group_names = sorted(list(set(labels)))
    group_subtables = [[] for _ in group_names]

    for row in table:
        group_by_val = row[group_by_col_index]
        subtable_index = group_names.index(group_by_val)
        group_subtables[subtable_index].append(row)
    return group_names, group_subtables

def get_column(table: list, index):
    col = [row[index] for row in table]
    return col

def compute_slope_intercept(x_list, y_list):
    '''compute the slope and y-intercept for the best fit line
        of the given data

        Args:
            x_list (list): data for the horizontal values.
            y_list (list): data for the vertical values.
        Returns:
            slope (int): the slope of the best fit line
            intercept (int): the y intercept of the best fit line
    '''
    meanx = np.mean(x_list)
    meany = np.mean(y_list)

    slope = sum([(x_list[i] - meanx) * (y_list[i] - meany) for i in range(len(x_list))]) / \
        sum([(x_list[i] - meanx) ** 2 for i in range(len(x_list))])

    # y = mx + b => b = y - mx
    intercept = meany - slope * meanx
    return slope, intercept

def calculate_covariance_and_coeff(x_list, y_list):
    '''compute the covariance and correlation coefficient for the best fit line
        of the given data

        Args:
            x_list (list): data for the horizontal values.
            y_list (list): data for the vertical values.
        Returns:
            cov (int): the covariance of the best fit line
            coeff (int): the correlation coefficient of the best fit line
    '''
    meanx = np.mean(x_list)
    meany = np.mean(y_list)

    cov = sum([(x_list[i] - meanx) * (y_list[i] - meany) for i in range(len(x_list))]) / len(x_list)
    coeff = cov / (np.std(x_list) * np.std(y_list))
    return cov, coeff